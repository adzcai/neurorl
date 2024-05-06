'''
// interactive session
salloc -p test -t 0-01:00 --mem=200000 

salloc -p gpu_test -t 0-01:00 --mem=8000 --gres=gpu:1

module load python/3.10.12-fasrc01
mamba activate neurorl

// launch parallel job
python configs/language/train_muzero.py \
  --search='initial' \
  --parallel='sbatch' \
  --num_actors=1 \
  --use_wandb=True \
  --partition=gpu \
  --wandb_entity=yichenli \
  --wandb_project=language \
  --run_distributed=True \
  --time=0-72:00:00 

// test in interactive session
python configs/language/train_muzero.py \
  --search='initial' \
  --parallel='none' \
  --run_distributed=True \
  --debug=True \
  --use_wandb=False 
'''
import functools 
from typing import Dict

import dataclasses
from absl import flags # absl for app configurations (distributed commandline flags, custom logging modules)
from absl import app
from absl import logging
import os
from ray import tune
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme.jax import networks as networks_lib
from acme.jax.networks import duelling
from acme import wrappers as acme_wrappers
from acme import specs
from acme.jax import experiments
import gymnasium
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import rlax
import wandb
import matplotlib.pyplot as plt

from td_agents import basics, muzero
from library import muzero_mlps

from library.dm_env_wrappers import GymWrapper
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils
import library.networks as networks

from envs.language import langenv
from envs.language.cfg import configurations 

obsfreq = 5000 # frequency to call observer
plotfreq = 0.1 #50000 # frequency to plot action trajectory
UP_PRESSURE_THRESHOLD = 5 # pressure threshold to increase curriculum
DOWN_PRESSURE_THRESHOLD = 10 # pressure threshold to decrease curriculum
UP_REWARD_THRESHOLD = 0.7 # upper reward threshold for incrementing up pressure
DOWN_REWARD_THRESHOLD = -5 #0.5 # lower reward threshold for incrementing down pressure
up_pressure = 0 # initial up pressure
down_pressure = 0 # initial down pressure


# -----------------------
# command line flags definition, using absl library
# -----------------------
flags.DEFINE_string('config_file', '', 'config file') # ('flag name', 'default value', 'value interpretation')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')
# organize all flags 
FLAGS = flags.FLAGS
# more flags are in parallel.py and experiment_builder.py


State = jax.Array


def observation_encoder(
    inputs: acme_wrappers.observation_action_reward.OAR,
    num_actions: int,
    max_sentence_length: int=configurations['max_sentence_length'],
    num_fibers: int=configurations['num_fibers'],
    num_areas: int=configurations['num_areas'],
    max_assemblies: int=configurations['max_assemblies'],
    num_pos: int=configurations['num_pos'],
    num_words: int=configurations['num_words'],):
  """
  A neural network to encode the environment observation / state.
  In the case of parsing blocks, 
    it creates embeddings for different stacks, pointer info, 
    and embeddings for previous reward and action,
    then it concatenates all embeddings as input.
  The neural network is a multi-layer perceptron with relu.
	observation: [cur lex readout (initialized as all -1s),
					goal lex (padding with -1),
					goal part of speech (padding -1 at the end),
					fiber inhibition status (initialized as all closed 0s),
					area inhibition status (initialized as all closed 0s),
					last activated assembly idx in the area (initialized as all -1s), 
					number of lexicon-connected assemblies in each area (initialized as 0s, or max_lexicon for lexicon area),
					number of all assemblies in each area (initialized as 0s, or max_lexicon for lexicon area),
					]
  Returns:
    The output of the neural network, ie. the encoded representation.
  """


  # embeddings for different elements in state repr
  curlex_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut1 = max_sentence_length # state idx as cutting point
  goallex_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut2 = cut1+max_sentence_length
  goalpos_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut3 = cut2+max_sentence_length
  fiber_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut4 = cut3+num_fibers
  area_embed1 = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())
  cut5 = cut4+num_areas
  area_embed2 = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut6 = cut5+num_areas
  area_embed3 = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut7 = cut6+num_areas
  area_embed4 = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  # embeddings for prev reward and action
  reward_embed = hk.Linear(128, w_init=hk.initializers.RandomNormal())
  action_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  # GRU for goallex and goalpos
  gru = hk.GRU(512, w_i_init=hk.initializers.TruncatedNormal(), b_init=hk.initializers.TruncatedNormal())
  gru_out_embed = hk.Linear(512, w_init=hk.initializers.TruncatedNormal())
  # backbone of the encoder
  mlp = hk.nets.MLP([512,512,512,512], activate_final=True) # default RELU activations between layers (and after final layer)
  def fn(x, dropout_rate=None):
    # process gru
    gru_input = jnp.concatenate((
                        goallex_embed(jax.nn.one_hot(x.observation[cut1:cut2], num_words)),
                        goalpos_embed(jax.nn.one_hot(x.observation[cut2:cut3], num_pos)),
                        ), axis=1)
    state = gru.initial_state(None)
    output_sequence, final_state = hk.static_unroll(core=gru, input_sequence=gru_input, initial_state=state, time_major=True)
    goal_repr = jnp.sum(output_sequence, axis=0) # sum along time dimension
    print(f"gru_input shape {gru_input.shape} \noutput_sequence shape {output_sequence.shape} \ngoal_repr shape {goal_repr.shape}")
    # concatenate embeddings and previous reward and action
    x = jnp.concatenate((
        curlex_embed(jax.nn.one_hot(x.observation[:cut1], num_words).reshape(-1)),
        # goallex_embed(jax.nn.one_hot(x.observation[cut1:cut2], num_words).reshape(-1)),
        # goalpos_embed(jax.nn.one_hot(x.observation[cut2:cut3], num_pos).reshape(-1)),
        gru_out_embed(goal_repr.reshape(-1)),
        fiber_embed(jax.nn.one_hot(x.observation[cut3:cut4], 2).reshape(-1)),
        area_embed1(jax.nn.one_hot(x.observation[cut4:cut5], 2).reshape(-1)),
        area_embed2(jax.nn.one_hot(x.observation[cut5:cut6], max_assemblies).reshape(-1)),
        area_embed3(jax.nn.one_hot(x.observation[cut6:cut7], max_assemblies).reshape(-1)),
        area_embed4(jax.nn.one_hot(x.observation[cut7:], max_assemblies).reshape(-1)),
        reward_embed(jnp.expand_dims(x.reward, 0)), 
        action_embed(jax.nn.one_hot(x.action, num_actions))  
      ))
    # relu first, then mlp, relu
    x = jax.nn.relu(x)
    x = mlp(x, dropout_rate=dropout_rate)
    return x
  # If there's a batch dim, applies vmap first.
  has_batch_dim = inputs.reward.ndim > 0
  if has_batch_dim: # have batch dimension
    fn = jax.vmap(fn)
  return fn(inputs)


def make_muzero_networks(
    env_spec: specs.EnvironmentSpec,
    config: muzero.Config,
    **kwargs) -> muzero.MuZeroNetworks:
  num_actions = int(env_spec.actions.maximum - env_spec.actions.minimum) + 1
  def make_core_module() -> muzero.MuZeroNetworks:
    observation_fn = functools.partial(observation_encoder,
                                       num_actions=num_actions)
    observation_fn = hk.to_module(observation_fn)('obs_fn')
    state_fn = networks.DummyRNN()
    def transition_fn(action: int, state: State):
      # Setup transition function: ResNet. action: [A], state: [D]
      action_onehot = jax.nn.one_hot(
          action, num_classes=num_actions)
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"
      def _transition_fn(action_onehot, state):
        """ResNet transition model that scales gradient."""
        out = muzero_mlps.SimpleTransition(
            num_blocks=config.transition_blocks)(
            action_onehot, state)
        out = muzero.scale_gradient(out, config.scale_grad)
        return out, out
      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)
    transition_fn = hk.to_module(transition_fn)('transition_fn')
    # Setup prediction functions: policy, value, reward
    root_value_fn = hk.nets.MLP(
        (128, 32, config.num_bins), name='pred_root_value')
    root_policy_fn = hk.nets.MLP(
        (128, 32, num_actions), name='pred_root_policy')
    model_reward_fn = hk.nets.MLP(
        (32, 32, config.num_bins), name='pred_model_reward')

    if config.seperate_model_nets: # what is typically done
      model_value_fn = hk.nets.MLP(
          (128, 32, config.num_bins), name='pred_model_value')
      model_policy_fn = hk.nets.MLP(
          (128, 32, num_actions), name='pred_model_policy')
    else:
      model_value_fn = root_value_fn
      model_policy_fn = root_policy_fn
    def root_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"
      def _root_predictor(state: State):
        policy_logits = root_policy_fn(state)
        value_logits = root_value_fn(state)
        return muzero.RootOutput(
            state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
        )
      if state.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)
    def model_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"
      def _model_predictor(state: State):
        reward_logits = model_reward_fn(state)
        policy_logits = model_policy_fn(state)
        value_logits = model_value_fn(state)
        return muzero.ModelOutput(
            new_state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
            reward_logits=reward_logits,
        )
      if state.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)
    return muzero.MuZeroArch(
        observation_fn=observation_fn,
        state_fn=state_fn,
        transition_fn=transition_fn,
        root_pred_fn=root_predictor,
        model_pred_fn=model_predictor)
  return muzero.make_network(
    environment_spec=env_spec,
    make_core_module=make_core_module,
    **kwargs)
  

class MuObserver(basics.ActorObserver):
  """
  An observer for tracking actions, rewards, and states during experiment.
  Log observed information and visualizations to wandb.
  """
  def __init__(self,
               period: int = obsfreq,
               prefix: str = 'MuObserver',
               plot_every: int = plotfreq):
    super(MuObserver, self).__init__()
    self.period = period
    self.prefix = prefix
    self.idx = -1
    self.logging = True
    self.plot_every = plot_every
  def wandb_log(self, d: dict):
    if self.logging:
      if wandb.run is not None:
        wandb.log(d)
      else:
        self.logging = False
        self.period = np.inf
  def observe_first(self, state: basics.ActorState, timestep: dm_env.TimeStep) -> None:
    """Observes the initial state and initial time-step.
    Usually state will be all zeros and time-step will be output of reset."""
    self.idx += 1
    # epsiode just ended, flush metrics if you want
    if self.idx > 0:
      self.get_metrics()
    # start collecting metrics again
    self.actor_states = [state]
    self.timesteps = [timestep]
    self.actions = []
  def observe_action(self, state: basics.ActorState, action: jax.Array) -> None:
    """Observe state and action that are due to observation of time-step.
    Should be state after previous time-step along"""
    self.actor_states.append(state)
    self.actions.append(action)
  def observe_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Observe next.
    Should be time-step after selecting action"""
    self.timesteps.append(timestep)

  def get_metrics(self) -> Dict[str, any]:
    """Returns metrics collected for the current episode."""
    if self.idx==0 or (not self.idx % self.period == 0):
      return
    if not self.logging:
      return 
    print('\n\nlogging!')
    import envs.language.cfg as langcfg
    max_steps = langcfg.configurations['max_steps']
    curriculum = langcfg.configurations['curriculum']
    action_dict = langcfg.configurations['action_dict']
    global up_pressure, down_pressure, UP_PRESSURE_THRESHOLD, DOWN_PRESSURE_THRESHOLD, UP_REWARD_THRESHOLD, DOWN_REWARD_THRESHOLD
    tmp_down_threshold = DOWN_PRESSURE_THRESHOLD * (curriculum-1) if 2<=curriculum<=configurations['max_complexity'] else DOWN_PRESSURE_THRESHOLD*configurations['max_sentence_length'] # adjust threshold for higher curriculum
    print(f"current curriculum {curriculum}, up_pressure {up_pressure} / {UP_PRESSURE_THRESHOLD}, down_pressure {down_pressure} / {tmp_down_threshold}")
    # first prediction is empty (None)
    results = {}
    npreds = len(self.actor_states[1:])
    actions = jnp.stack(self.actions)[:npreds]
    action_names = [action_dict[a.item()] for a in actions]
    rewards = jnp.stack([t.reward for t in self.timesteps[1:]])
    observations = jnp.stack([t.observation.observation for t in self.timesteps[1:]])
    # current episode reward
    episode_reward = jnp.sum(rewards)
    results['episode_reward'] = episode_reward 
    # log the metrics
    results["actions"] = actions
    results["action_names"] = action_names
    results["rewards"] = rewards
    results["observations"] = observations 
    # plot actions
    if self.idx % self.plot_every == 0: 
      fig, ax = plt.subplots(max_steps//10, 10, figsize=(30, 6*(max_steps//10)))
      cut1 = max_sentence_length # state idx as cutting point
      cut2 = cut1+max_sentence_length
      cut3 = cut2+max_sentence_length
      cut4 = cut3+num_fibers
      cut5 = cut4+num_areas
      cut6 = cut5+num_areas
      cut7 = cut6+num_areas
      for t in range(npreds):
        irow = t//10
        jcol = t%10
        ax[irow,jcol].axhline(y=19, xmin=0, xmax=10)
        ax[irow,jcol].text(0.2, 13.5, f"Curr:\n{observations[t][:cut1].reshape(configurations['puzzle_max_stacks'], configurations['stack_max_blocks'])}", 
                          style='italic', bbox={'facecolor': 'orange', 'alpha': 0.2, 'pad': 1})
        ax[irow,jcol].text(0.2, 8, f"Goal:\n{observations[t][cut1:cut2].reshape(configurations['puzzle_max_stacks'], configurations['stack_max_blocks'])}", 
                          style='italic', bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 1})
        ax[irow,jcol].text(0.2, 5, f"Table:\n{observations[t][cut2:cut3].reshape(min(2, configurations['puzzle_max_blocks']//7),-1)}", 
                          style='italic', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 1})
        ax[irow,jcol].text(0.2, 3, f"Correct: {observations[t][cut3:cut4]}", 
                          style='italic', bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 1})
        ax[irow,jcol].text(0.2, 2, f"Spointer: {observations[t][cut4:cut5]}, Tpointer: {observations[t][cut5:cut6]}", 
                          style='italic', bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 1})
        ax[irow,jcol].text(0.2, 1, f"Iparsed: {observations[t][cut6:cut7]}, Gparsed: {observations[t][cut7:]}", 
                          style='italic', bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 1})
        ax[irow,jcol].set_xticks([])
        ax[irow,jcol].set_yticks([])
        ax[irow,jcol].set_ylim(0,19.1)
        ax[irow,jcol].set_xlim(0,10.1)
        ax[irow,jcol].set_title(f"A={action_names[t]}\nR={round(float(rewards[t]),5)}")
      plt.suptitle(t=f"Curriculum {curriculum}, up_pressure {up_pressure} / {UP_PRESSURE_THRESHOLD}, down_pressure {down_pressure} / {tmp_down_threshold}\nepisode reward={episode_reward}",
                    x=0.5, y=0.89)
      self.wandb_log({f"{self.prefix}/trajectory": wandb.Image(fig)})
      plt.close(fig)
    # print first 50 actions
    for t in range(min(npreds, 50)):
      print(f"t={t}, A={action_names[t]}, R={round(float(rewards[t]),5)}, state={observations[t]}")
    print(f'\ncurrent episode rewards {episode_reward}')
    # check curriculum
    if episode_reward > UP_REWARD_THRESHOLD:
      up_pressure += 1
      down_pressure = 0
      print(f"up_pressure + 1 = {up_pressure} / {UP_PRESSURE_THRESHOLD}")
    elif episode_reward < DOWN_REWARD_THRESHOLD:
      down_pressure += 1
      up_pressure = 0
      print(f"down_pressure + 1 = {down_pressure} / {tmp_down_threshold}")
    else: # reset up and down pressure
      up_pressure = 0
      down_pressure = 0
    # up pressure reached threshold
    if up_pressure >= UP_PRESSURE_THRESHOLD: 
      if 2<=curriculum<=configurations['max_complexity']-1:
        curriculum += 1
        print(f'up_pressure reached threshold, increasing curriculum from {curriculum-1} to {curriculum}')
      elif curriculum==configurations['max_complexity']:
        curriculum = configurations['max_complexity'] 
        print(f"up_pressure reached threshold, staying at curriculum {curriculum}")
      elif curriculum==0: 
        print(f'up_pressure reached threshold, staying at curriculum {curriculum}')
      else:
        raise ValueError(f"curriculum {curriculum} should be int in set(0, 2, 3, ..., {configurations['max_complexity']})")
      up_pressure = 0 # release pressure
    # down pressure reached threshold
    elif down_pressure >= tmp_down_threshold: 
      if 3<=curriculum<=configurations['max_complexity']:
        curriculum -= 1
        print(f'down_pressure reached threshold, decreasing curriculum from {curriculum+1} to {curriculum}')
      elif curriculum==0:
        print(f"down_pressure reached threshold, staying at curriculum {curriculum}")
      elif curriculum==2:
        curriculum = 2
        print(f"down_pressure reached threshold, staying at curriculum {curriculum}")
      else:
        raise ValueError(f"curriculum {curriculum} should be int in set(0, 2, 3, ..., {configurations['max_complexity']})")
      down_pressure = 0 # release pressure
    langcfg.configurations['curriculum'] = curriculum # update curriculum in cfg file
    print('logging ends\n\n')
    return results
  

def make_environment(seed: int = 0 ,
                     evaluation: bool = False,
                     **kwargs) -> dm_env.Environment:
  """
  Initializes and wraps the environment simulator with specific settings.
  Returns:
    dm_env.Environment object
      with multiple elements wrapped together (simulator, observation, action, reward, single precision).
  """
  del seed
  del evaluation

  # create dm_env
  sim = langenv.Simulator()
  sim.reset()
  
  # insert info into cfg
  import envs.language.cfg as langcfg
  langcfg.configurations['num_actions'] = sim.num_actions
  langcfg.configurations['action_dict'] = sim.action_dict
  langcfg.configurations['max_sentence_length'] = sim.max_sentence_length
  langcfg.configurations['num_areas'] = sim.num_areas
  langcfg.configurations['num_fibers'] = sim.num_fibers
  env = langenv.EnvWrapper(sim)

  # add acme wrappers
  wrapper_list = [ # put action + reward in observation
    acme_wrappers.ObservationActionRewardWrapper,# cheaper to do computation in single precision
    acme_wrappers.SinglePrecisionWrapper,
  ]

  return acme_wrappers.wrap_all(env, wrapper_list)


def setup_experiment_inputs(
    agent_config_kwargs: dict=None,
    env_kwargs: dict=None,
    debug: bool = False,
  ):
  """
  Prepares inputs for experiments,
    including agent configs, env settings, and debugging options.

  Returns a OnlineExperimentConfigInputs object (in library/experiment_builder.py)
    consist of a named tuple with settings such as agent name, agent config, env factory, observers, etc.
  """
  config_kwargs = agent_config_kwargs or dict()
  env_kwargs = env_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  agent = agent_config_kwargs.get('agent', '')
  assert agent != '', 'please set agent'

  if agent == 'muzero':
    config = muzero.Config(**config_kwargs)
    import mctx
    # currently using same policy in learning and acting
    mcts_policy = functools.partial(
      mctx.gumbel_muzero_policy,
      max_depth=config.max_sim_depth,
      num_simulations=config.num_simulations,
      gumbel_scale=config.gumbel_scale)

    discretizer = utils.Discretizer(
                  num_bins=config.num_bins,
                  max_value=config.max_scalar_value,
                  tx_pair=config.tx_pair,
              )

    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
          muzero.get_actor_core,
          mcts_policy=mcts_policy,
          discretizer=discretizer,
      ),
      ActorCls=functools.partial(
        basics.BasicActor,
        observers=[MuObserver(period=1 if debug else obsfreq)],
        ),
      optimizer_cnstr=muzero.muzero_optimizer_constr,
      LossFn=muzero.MuZeroLossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
          discretizer=discretizer,
          mcts_policy=mcts_policy,
          simulation_steps=config.simulation_steps,
          #reanalyze_ratio=config.reanalyze_ratio,
          # reanalyze_ratio=0.25, # how frequently to tune value loss wrt tree node values (faster but less accurate)
          reanalyze_ratio=0.1,
          root_policy_coef=config.root_policy_coef,
          root_value_coef=config.root_value_coef,
          model_policy_coef=config.model_policy_coef,
          model_value_coef=config.model_value_coef,
          model_reward_coef=config.model_reward_coef,
      ))
    network_factory = functools.partial(make_muzero_networks, config=config)
  else:
    raise NotImplementedError(agent)
  # load environment factory
  environment_factory = functools.partial(
    make_environment,
    **env_kwargs)
  # setup observer factory for environment
  # this logs the average every reset=50 episodes (instead of every episode)
  observers = [
      utils.LevelAvgReturnObserver(
        reset=50,
        get_task_name=lambda e: "task"
        ),
      ]
  return experiment_builder.OnlineExperimentConfigInputs(
    agent=agent,
    agent_config=config,
    final_env_kwargs=env_kwargs,
    builder=builder,
    network_factory=network_factory,
    environment_factory=environment_factory,
    observers=observers,
  )


def train_single(
    env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    log_dir: str = None,
    num_actors: int = 1,
    run_distributed: bool = False,
):
  """
  Function for running individual training experiment.
  Set up logging, environment, and agent.
  """
  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    agent_config_kwargs=agent_config_kwargs,
    env_kwargs=env_kwargs,
    debug=debug) # Returns a OnlineExperimentConfigInputs object (in library/experiment_builder.py), 
                  # Consist of a named tuple with settings such as agent name, agent config, env factory, observers, etc.

  logger_factory_kwargs = dict(
    actor_label="actor",
    evaluator_label="evaluator",
    learner_label="learner",
  )

  experiment = experiment_builder.build_online_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    logger_factory_kwargs=logger_factory_kwargs,
    debug=debug
  ) # Returns acme.jax.experiments.ExperimentConfig object, 
      # which contains information about networks, evaluator, observers, env, logger, checkpoint etc.
      # The class has a callable function for evaluator factory function.
      # Source code: https://github.com/google-deepmind/acme/blob/master/acme/jax/experiments/config.py#L123

  config = experiment_config_inputs.agent_config # Retrieves the agent config dictionary

  if run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=experiment,
        num_actors=num_actors) # Calls a function in acme.jax.experiments 
                                  # Returns a Launchpad program with all nodes needed for running distributed experiment,
                                  # Nodes include actors, learners, inference servers, etc.
                                  # Source code: https://github.com/google-deepmind/acme/blob/master/acme/jax/experiments/make_distributed_experiment.py
    local_resources = {
        "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
    } # run non-compute-intensive tasks on CPU in distributed way to save GPU resources
    controller = lp.launch(program,
                           lp.LaunchType.LOCAL_MULTI_PROCESSING,
                           terminal='current_terminal',
                           local_resources=local_resources) # Launches the distributed experiment using Launchpad 
                                                              # Source code: https://github.com/google-deepmind/launchpad/blob/3b28eaed02c4294197b9ca2b8988cf68d8b5d868/launchpad/launch/local_multi_processing/launch.py
    controller.wait(return_on_first_completed=True) # Waits for the first component to complete/failure as a trigger to end the experiment
                                                    # Source code: https://github.com/google-deepmind/launchpad/blob/3b28eaed02c4294197b9ca2b8988cf68d8b5d868/launchpad/launch/worker_manager.py#L412
    controller._kill() # Then terminates the experiment (all lp nodes)
                          # Source code: https://github.com/google-deepmind/launchpad/blob/3b28eaed02c4294197b9ca2b8988cf68d8b5d868/launchpad/launch/worker_manager.py#L318
  else:
    experiments.run_experiment(experiment=experiment) # Runs a single-threaded training loop
                                                        # Source code: https://github.com/google-deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py

def setup_wandb_init_kwargs():
  if not FLAGS.use_wandb:
    return dict()

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      name=FLAGS.wandb_name,
      group=FLAGS.search,
      save_code=False,
  )
  return wandb_init_kwargs

def run_single():
  ########################
  # default settings
  ########################
  env_kwargs = dict()
  agent_config_kwargs = dict()
  num_actors = FLAGS.num_actors
  run_distributed = FLAGS.run_distributed
  wandb_init_kwargs = setup_wandb_init_kwargs()
  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      samples_per_insert=1.0,
      min_replay_size=100,
    ))
    env_kwargs.update(dict(
    ))

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results'

  if FLAGS.make_path: # default False from parallel slurm jobs
    # i.e. ${folder}/runs/${date_time}/
    folder = parallel.gen_log_dir(
        base_dir=os.path.join(folder, 'rl_results'),
        hourminute=True,
        date=True,
    )

  ########################
  # override with config settings, e.g. from parallel run
  ########################
  if FLAGS.config_file: # parallel run should pass in a config_file that's created online/temporarily
    configs = utils.load_config(FLAGS.config_file)
    config = configs[FLAGS.config_idx-1]  # FLAGS.config_idx starts at 1 with SLURM
    logging.info(f'loaded config: {str(config)}')

    agent_config_kwargs.update(config['agent_config'])
    env_kwargs.update(config['env_config'])
    folder = config['folder']

    num_actors = config['num_actors'] # default 6 from parallel slurm jobs
    run_distributed = config['run_distributed'] # default True from parallel slurm jobs

    wandb_init_kwargs['group'] = config['wandb_group']
    wandb_init_kwargs['name'] = config['wandb_name']
    wandb_init_kwargs['project'] = config['wandb_project']
    wandb_init_kwargs['entity'] = config['wandb_entity']

    if not config['use_wandb']:
      wandb_init_kwargs = dict()


  if FLAGS.debug and not FLAGS.subprocess: # FLAGS.subprocess default True from parallel slurm jobs
      configs = parallel.get_all_configurations(spaces=sweep(FLAGS.search))
      first_agent_config, first_env_config = parallel.get_agent_env_configs(
          config=configs[0])
      agent_config_kwargs.update(first_agent_config)
      env_kwargs.update(first_env_config)

  if not run_distributed:
    assert agent_config_kwargs['samples_per_insert'] > 0

  train_single(
    wandb_init_kwargs=wandb_init_kwargs,
    env_kwargs=env_kwargs,
    agent_config_kwargs=agent_config_kwargs,
    log_dir=folder,
    num_actors=num_actors,
    run_distributed=run_distributed
    )

def run_many():
  wandb_init_kwargs = setup_wandb_init_kwargs() # group will be 'FLAGS.search' by default

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results_dir'

  assert FLAGS.debug is False, 'only run debug if not running many things in parallel'
  # and FLAGS.parallel should be 'none' for debug
  if FLAGS.parallel == 'ray':
    parallel.run_ray(
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      debug=FLAGS.debug,
      folder=folder,
      space=sweep(FLAGS.search),
      make_program_command=functools.partial(
        parallel.make_program_command,
        trainer_filename=__file__,
        run_distributed=FLAGS.run_distributed,
        num_actors=FLAGS.num_actors),
    )
  elif FLAGS.parallel == 'sbatch': # fasrc is sbatch system
    # this will submit multiple sbatch jobs, each will call run_single(distributed=True)
    parallel.run_sbatch(
      trainer_filename=__file__,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      folder=folder,
      run_distributed=FLAGS.run_distributed, # usually user command will set this to True if parallel
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep(FLAGS.search), # usually search is 'initial'
      num_actors=FLAGS.num_actors) # default flag is 6 (in parallel.py)

def sweep(search: str = 'default'):
  if search == 'initial':
    space = [
        {
            "group": tune.grid_search(['Mgru2+comp-5v.7nospace']),
            "num_steps": tune.grid_search([500e6]),

            "samples_per_insert": tune.grid_search([20.0]),
            "batch_size": tune.grid_search([128]),
            "trace_length": tune.grid_search([10]),
            "learning_rate": tune.grid_search([1e-3]),

            "agent": tune.grid_search(['muzero']),
            "num_bins": tune.grid_search([1001]),  # for muzero
            "num_simulations": tune.grid_search([10]), # for muzero

            "state_dim": tune.grid_search([1024]),
        }
    ]
  else:
    raise NotImplementedError(search)
  return space

def main(_):
  assert FLAGS.parallel in ('ray', 'sbatch', 'none')
  if FLAGS.parallel in ('ray', 'sbatch'):
    run_many()
  else:
    run_single()

if __name__ == '__main__':
  app.run(main)
