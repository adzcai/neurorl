'''
// interactive session
salloc -p gpu_test -t 0-01:00 --mem=8000 --gres=gpu:1
module load python/3.10.12-fasrc01
mamba activate neurorl

// launch parallel job
python configs/plan_trainer.py \
  --search='initial' \
  --parallel='sbatch' \
  --num_actors=1 \
  --use_wandb=True \
  --partition=gpu \
  --wandb_entity=yichenli \
  --wandb_project=plan \
  --run_distributed=True \
  --time=0-72:00:00 

// test in interactive session
python configs/plan_trainer.py \
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

from td_agents import q_learning, basics, muzero
from library import muzero_mlps

from library.dm_env_wrappers import GymWrapper
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils
import library.networks as networks

from envs.blocksworld import plan
from envs.blocksworld.cfg import configurations 

obsfreq = 5000 # frequency to call observer
plotfreq = 50000 # frequency to plot action trajectory
UP_PRESSURE_THRESHOLD = 5 # pressure threshold to increase curriculum
DOWN_PRESSURE_THRESHOLD = 10 # pressure threshold to decrease curriculum
UP_REWARD_THRESHOLD = 0.8 # upper reward threshold for incrementing up pressure
DOWN_REWARD_THRESHOLD = -2 #0.5 # lower reward threshold for incrementing down pressure
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
    stack_max_blocks: int=configurations['stack_max_blocks'],
    puzzle_max_blocks: int=configurations['puzzle_max_blocks'],
    puzzle_max_stacks: int=configurations['puzzle_max_stacks']):
  """
  A neural network to encode the environment observation / state.
  In the case of parsing blocks, 
    it creates embeddings for different stacks, pointer info, 
    and embeddings for previous reward and action,
    then it concatenates all embeddings as input.
  The neural network is a multi-layer perceptron with relu.

  Returns:
    The output of the neural network, ie. the encoded representation.
  """
  # embeddings for different elements in state repr
  curstack_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut1 = stack_max_blocks*puzzle_max_stacks # state idx as cutting point
  goalstack_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut2 = cut1+stack_max_blocks*puzzle_max_stacks
  table_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
  cut3 = cut2+puzzle_max_blocks
  corhist_embed = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())
  cut4 = cut3+puzzle_max_stacks
  spointer_embed = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())
  cut5 = cut4+1
  tpointer_embed = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())
  cut6 = cut5+1
  iparsed_embed = hk.Linear(64, w_init=hk.initializers.TruncatedNormal())
  cut7 = cut6+1
  gparsed_embed = hk.Linear(64, w_init=hk.initializers.TruncatedNormal())
  # embeddings for prev reward and action
  reward_embed = hk.Linear(128, w_init=hk.initializers.RandomNormal())
  action_embed = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())
  # backbone of the encoder: mlp with relu
  mlp = hk.nets.MLP([512,512,512,512], activate_final=True) # default RELU activations between layers (and after final layer)
  def fn(x, dropout_rate=None):
    # concatenate embeddings and previous reward and action
    x = jnp.concatenate((
        curstack_embed(jax.nn.one_hot(x.observation[:cut1], puzzle_max_blocks).reshape(-1)),
        goalstack_embed(jax.nn.one_hot(x.observation[cut1:cut2], puzzle_max_blocks).reshape(-1)),
        table_embed(jax.nn.one_hot(x.observation[cut2:cut3], puzzle_max_blocks).reshape(-1)),
        corhist_embed(jax.nn.one_hot(x.observation[cut3:cut4], stack_max_blocks).reshape(-1)),
        spointer_embed(jax.nn.one_hot(x.observation[cut4:cut5], puzzle_max_stacks).reshape(-1)),
        tpointer_embed(jax.nn.one_hot(x.observation[cut5:cut6], puzzle_max_blocks).reshape(-1)),
        iparsed_embed(jax.nn.one_hot(x.observation[cut6:cut7], 2).reshape(-1)),
        iparsed_embed(jax.nn.one_hot(x.observation[cut7:], 2).reshape(-1)),
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


def make_qlearning_networks(
        env_spec: specs.EnvironmentSpec,
        config: q_learning.Config,
        ):
  """
  Builds default R2D2 networks for Q-learning based on the environment specifications and configurations.
  """
  num_actions = int(env_spec.actions.maximum - env_spec.actions.minimum) + 1
  import envs.blocksworld.cfg as bwcfg
  assert num_actions == bwcfg.configurations['plan']['num_actions']

  def make_core_module() -> q_learning.R2D2Arch:

    observation_fn = functools.partial(
      observation_encoder, 
      num_actions=num_actions)
    return q_learning.R2D2Arch(
      torso=hk.to_module(observation_fn)('obs_fn'),
      memory=networks.DummyRNN(),  # nothing happens
      head=duelling.DuellingMLP(num_actions,
                                hidden_sizes=[config.q_dim]))

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)


class QObserver(basics.ActorObserver):
  """
  An observer for tracking actions, rewards, and states during experiment.
  Log observed information and visualizations to wandb.
  """
  def __init__(self,
               period: int = obsfreq,
               prefix: str = 'QObserver',
               plot_every: int = plotfreq):
    super(QObserver, self).__init__()
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
    import envs.blocksworld.cfg as bwcfg
    max_steps = bwcfg.configurations['plan']['max_steps']
    curriculum = bwcfg.configurations['plan']['curriculum']
    global up_pressure, down_pressure, UP_PRESSURE_THRESHOLD, DOWN_PRESSURE_THRESHOLD, UP_REWARD_THRESHOLD, DOWN_REWARD_THRESHOLD
    tmp_down_threshold = DOWN_PRESSURE_THRESHOLD * (curriculum-1) if 2<=curriculum<=configurations['puzzle_max_blocks'] else DOWN_PRESSURE_THRESHOLD*configurations['puzzle_max_blocks'] # adjust threshold for higher curriculum
    print(f"current curriculum {curriculum}, up_pressure {up_pressure} / {UP_PRESSURE_THRESHOLD}, down_pressure {down_pressure} / {tmp_down_threshold}")
    # first prediction is empty (None)
    results = {}
    action_dict = bwcfg.configurations['plan']['action_dict']
    q_values = [s.predictions for s in self.actor_states[1:]]
    q_values = jnp.stack(q_values)
    npreds = len(q_values)
    actions = jnp.stack(self.actions)[:npreds]
    q_values = rlax.batched_index(q_values, actions)
    action_names = [action_dict[a.item()] for a in actions]
    rewards = jnp.stack([t.reward for t in self.timesteps[1:]])
    observations = jnp.stack([t.observation.observation for t in self.timesteps[1:]])
    # current episode reward
    episode_reward = jnp.sum(rewards)
    results['episode_reward'] = episode_reward 
    # log the metrics
    results["actions"] = actions
    results["action_names"] = action_names
    results["q_values"] = q_values
    results["rewards"] = rewards
    results["observations"] = observations 
    # plot actions
    if self.idx % self.plot_every == 0: 
      fig, ax = plt.subplots(max_steps//10, 10, figsize=(30, 6*(max_steps//10)))
      cut1 = configurations['stack_max_blocks']*configurations['puzzle_max_stacks'] # state idx as cutting point
      cut2 = cut1+configurations['stack_max_blocks']*configurations['puzzle_max_stacks']
      cut3 = cut2+configurations['puzzle_max_blocks']
      cut4 = cut3+configurations['puzzle_max_stacks']
      cut5 = cut4+1
      cut6 = cut5+1
      cut7 = cut6+1
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
        ax[irow,jcol].set_title(f"A={action_names[t]}\nR={round(float(rewards[t]),5)}\nQ={round(float(q_values[t]),5)}")
      plt.suptitle(t=f"Curriculum {curriculum}, up_pressure {up_pressure} / {UP_PRESSURE_THRESHOLD}, down_pressure {down_pressure} / {tmp_down_threshold}\nepisode reward={episode_reward}",
                    x=0.5, y=0.89)
      self.wandb_log({f"{self.prefix}/trajectory": wandb.Image(fig)})
      plt.close(fig)
    # print first 50 actions
    for t in range(min(npreds, 50)):
      print(f"t={t}, A={action_names[t]}, R={round(float(rewards[t]),5)}, Q={round(float(q_values[t]),5)}, state={observations[t]}")
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
      if 2<=curriculum<=configurations['puzzle_max_blocks']-1:
        curriculum += 1
        print(f'up_pressure reached threshold, increasing curriculum from {curriculum-1} to {curriculum}')
      elif curriculum==configurations['puzzle_max_blocks']:
        curriculum = configurations['puzzle_max_blocks'] 
        print(f"up_pressure reached threshold, staying at curriculum {curriculum}")
      elif curriculum==0: 
        print(f'up_pressure reached threshold, staying at curriculum {curriculum}')
      else:
        raise ValueError(f"curriculum {curriculum} should be int in set(0, 2, 3, ..., {configurations['puzzle_max_blocks']})")
      up_pressure = 0 # release pressure
    # down pressure reached threshold
    elif down_pressure >= tmp_down_threshold: 
      if 3<=curriculum<=configurations['puzzle_max_blocks']:
        curriculum -= 1
        print(f'down_pressure reached threshold, decreasing curriculum from {curriculum+1} to {curriculum}')
      elif curriculum==0:
        print(f"down_pressure reached threshold, staying at curriculum {curriculum}")
      elif curriculum==2:
        curriculum = 2
        print(f"down_pressure reached threshold, staying at curriculum {curriculum}")
      else:
        raise ValueError(f"curriculum {curriculum} should be int in set(0, 2, 3, ..., {configurations['puzzle_max_blocks']})")
      down_pressure = 0 # release pressure
    bwcfg.configurations['plan']['curriculum'] = curriculum # update curriculum in cfg file
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
  sim = plan.Simulator()
  sim.reset()
  
  # insert info into cfg
  import envs.blocksworld.cfg as bwcfg
  bwcfg.configurations['plan']['num_actions'] = sim.num_actions
  bwcfg.configurations['plan']['action_dict'] = sim.action_dict
  env = plan.EnvWrapper(sim)

  # add acme wrappers
  wrapper_list = [
    # put action + reward in observation
    acme_wrappers.ObservationActionRewardWrapper,
    # cheaper to do computation in single precision
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

  if agent == 'qlearning':
    config = q_learning.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      ActorCls=functools.partial(
        basics.BasicActor,
        observers=[QObserver(period=1 if debug else obsfreq)],
        ),
      LossFn=q_learning.R2D2LossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
      ))
    network_factory = functools.partial(make_qlearning_networks, config=config)
  else:
    raise NotImplementedError(agent)

  # -----------------------
  # load environment factory
  # -----------------------
  environment_factory = functools.partial(
    make_environment,
    **env_kwargs)

  # -----------------------
  # setup observer factory for environment
  # this logs the average every reset=50 episodes (instead of every episode)
  # -----------------------
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
            "group": tune.grid_search(['Qsparse3~10comp-2v8max1-11-7']),
            "num_steps": tune.grid_search([500e6]),

            "samples_per_insert": tune.grid_search([20.0]),
            "batch_size": tune.grid_search([128]),
            "trace_length": tune.grid_search([10]),
            "learning_rate": tune.grid_search([1e-3]),
            "agent": tune.grid_search(['qlearning']),
            "state_dim": tune.grid_search([1024]),
            "q_dim": tune.grid_search([1024]),
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
