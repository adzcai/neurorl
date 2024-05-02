from typing import NamedTuple, Any, Callable
import os
import logging

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# https://github.com/google/jax/issues/8302
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import os.path
from acme import specs
from acme.jax import experiments
from acme.utils import counting
from acme.tf import savers
import jax
from glob import glob
import pickle

from td_agents import basics

import random
import numpy as np
from scipy.stats import sem
from acme import wrappers as acme_wrappers
import dm_env
from envs.blocksworld import plan

import configs.plan_trainer_muzero as muzerotrainer 
import envs.blocksworld.cfg as bwcfg
from td_agents import muzero
import functools
import mctx
import library.utils as utils
import envs.blocksworld.utils as bwutils

import haiku as hk
import jax.numpy as jnp


def load_config(filename):
  with open(filename, 'rb') as fp:
    config = pickle.load(fp)
    logging.info(f'Loaded: {filename}')
    return config

class LoadOutputs(NamedTuple):
  config: Any
  builder: Any
  learner: Any
  policy: Any
  checkpointer: Any
  actor: Any


def load_agent(
    env,
    config: basics.Config,
    builder: basics.Builder,
    network_factory: Callable[
      [specs.EnvironmentSpec, basics.Config], basics.NetworkFn],
    seed_path: str = None,
    use_latest = True,
    evaluation = True,
    **kwargs):
  # then get environment spec
  environment_spec = specs.make_environment_spec(env)
  # the make network
  networks = network_factory(environment_spec)
  # make policy
  policy = builder.make_policy(
        networks=networks,
        environment_spec=environment_spec,
        evaluation=evaluation)
  # make learner
  key = jax.random.PRNGKey(config.seed)
  learner_key, key = jax.random.split(key)
  learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=None,
        logger_fn=lambda x: None,
        environment_spec=environment_spec,
        replay_client=None,
        counter=None)
  # create checkpointer
  parent_counter = counting.Counter(time_delta=0.)
  # get all directories from year
  checkpointer = None
  if seed_path:
    dirs, _ = get_dirs_ckpts(seed_path)
    checkpointing = experiments.CheckpointingConfig(
        directory=dirs[0],
        add_uid=False,
        max_to_keep=None,
    )
    checkpointer = savers.Checkpointer(
            objects_to_save={'learner': learner, 'counter': parent_counter},
            time_delta_minutes=checkpointing.time_delta_minutes,
            directory=checkpointing.directory,
            add_uid=checkpointing.add_uid,
            max_to_keep=checkpointing.max_to_keep,
            keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
            checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
        )
    reload(checkpointer, seed_path, use_latest)
  # make actor
  actor_key, key = jax.random.split(key)
  # will need a custom actor
  actor = builder.make_actor(
        actor_key, policy, environment_spec, variable_source=learner, adder=None)
  return LoadOutputs(
    config=config,
    builder=builder,
    learner=learner,
    policy=policy,
    actor=actor,
    checkpointer=checkpointer,
  )

def get_dirs_ckpts(seed_path: str):
  dirs = glob(os.path.join(seed_path, "checkpoints/learner"))
  assert len(dirs) > 0
  ckpts = glob(os.path.join(dirs[0], "*"))
  assert len(ckpts) > 0
  return dirs, ckpts

def reload(checkpointer, seed_path, use_latest: bool = True):
  # get all directories from year, load checkpoint
  _, ckpts = get_dirs_ckpts(seed_path)
  ckpts.sort()
  assert use_latest, 'need to implement otherwise'
  latest = ckpts[-1].split(".index")[0]
  ckpt_path = latest
  assert os.path.exists(f'{ckpt_path}.index')
  # print('loading', ckpt_path)
  status = checkpointer._checkpoint.restore(ckpt_path)


def load_settings(
  base_dir: str = None,
  run: str = None,
  seed_path: str = None):
  if seed_path is None: 
    assert base_dir is not None and run is not None, 'set values for finding path'
    seed_path = glob(os.path.join(base_dir, run, '*'))[0]
  config_file = os.path.join(seed_path, 'config.pkl')
  config = load_config(config_file)
  final_agent_config = config
  final_env_config = {}
  return seed_path, final_env_config, final_agent_config


def make_test_environment(
                        puzzle_max_stacks, 
                        puzzle_max_blocks, 
                        stack_max_blocks, 
                        sparse_reward,
                        evaluation, 
                        eval_puzzle_num_blocks,
                        compositional, 
                        compositional_eval,
                        compositional_type, 
                        compositional_holdout,
                        test_puzzle,
                        ):
  """
  Make plan env for testing
  Returns:
    dm_env.Environment object, with multiple elements wrapped together (simulator, observation, action, reward, single precision).
  """
  # create dm_env
  sim = plan.Simulator(puzzle_max_stacks=puzzle_max_stacks,
                      puzzle_max_blocks=puzzle_max_blocks,
                      stack_max_blocks=stack_max_blocks,
                      sparse_reward=sparse_reward,
                      evaluation=evaluation,
                      eval_puzzle_num_blocks=eval_puzzle_num_blocks,
                      compositional=compositional, compositional_eval=compositional_eval,
                      compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                      test_puzzle=test_puzzle)
  sim.reset()
  # insert info into cfg
  import envs.blocksworld.cfg as bwcfg
  bwcfg.configurations['plan']['num_actions'] = sim.num_actions
  bwcfg.configurations['plan']['action_dict'] = sim.action_dict
  env = plan.EnvWrapper(sim)
  # add acme wrappers
  wrapper_list = [
    acme_wrappers.ObservationActionRewardWrapper, # put action + reward in observation
    acme_wrappers.SinglePrecisionWrapper, # cheaper to do computation in single precision
  ]
  return acme_wrappers.wrap_all(env, wrapper_list), sim


def main(
        eval_lvls, lvls, 
        eval_blocks, fixed_num_stacks,
        puzzle_max_stacks, 
        puzzle_max_blocks, 
        stack_max_blocks, 
        sparse_reward,
        compositional, 
        compositional_eval, compositional_type, compositional_holdout,
        groupname, searchname='initial', 
        eval_test_puzzles=False,
        eval_num_stacks=False, fixed_num_blocks=None,
        eval_steps=False, max_oracle_steps=40, nsamples=30,
        nrepeats=100,
        ):
    '''
      lvls: list[int]
        list of puzzle_num_blocks to evaluate, should be within range [2, puzzle_max_blocks]
      compositional: bool
        if True, evaluating using compositional holdout
        if False, evaluating using specified lvl for puzzle_num_blocks
      compositional_type: {None, 'newblock', 'newconfig'}
      compositional_holdout: list
      groupname: str
      searchname: str
      eval_test_puzzles: bool
        whether to evaluate on test puzzles from jbrain
      nrepeats: int
        num of episode samples for random and expert demo
    '''
    default_log_dir = os.environ['RL_RESULTS_DIR']
    base_dir = os.path.join(
      default_log_dir,
      searchname,  # search name
      groupname  # group name
    )
    seed_path, env_kwargs, agent_kwargs = load_settings(      
      base_dir=base_dir,
      run='.')


    config = muzero.Config(**agent_kwargs)

    def tmp_obs_encoder(
        inputs: acme_wrappers.observation_action_reward.OAR,
        num_actions: int,
        stack_max_blocks: int=stack_max_blocks,
        puzzle_max_blocks: int=puzzle_max_blocks,
        puzzle_max_stacks: int=puzzle_max_stacks):
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

    muzerotrainer.observation_encoder = lambda inputs, num_actions: tmp_obs_encoder(inputs=inputs,num_actions=num_actions)


    if eval_lvls and len(lvls)>0:
      modelsolved = [] # ratio of puzzles solved
      modelsolvedsem = [] 
      oraclesolved = []
      oraclesolvedsem = []
      heuristicsolved = []
      heuristicsolvedsem = []
      randomsolved = []
      randomsolvedsem = []
      modelreward = [] # avg episode reward
      modelrewardsem = []
      oraclereward = []
      oraclerewardsem = []
      heuristicreward = []
      heuristicrewardsem = []
      randomreward = []
      randomrewardsem = []
      modelsteps = [] # num of steps for solving a puzzle 
      modelstepssem = []
      oraclesteps = []
      oraclestepssem = []
      heuristicsteps = [] # num of steps for solving a puzzle
      heuristicstepssem = []
      randomsteps = [] # num of steps for solving a puzzle
      randomstepssem = []
      for lvl in lvls:
        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=lvl,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=None,
                                        )
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
              evaluation=True,
              mcts_policy=mcts_policy,
              discretizer=discretizer,
          ),
          ActorCls=functools.partial(
            basics.BasicActor,
            observers=[muzerotrainer.MuObserver(period=100000)],
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
              reanalyze_ratio=0.25, 
              root_policy_coef=config.root_policy_coef,
              root_value_coef=config.root_value_coef,
              model_policy_coef=config.model_policy_coef,
              model_value_coef=config.model_value_coef,
              model_reward_coef=config.model_reward_coef,
          ))
        network_factory = functools.partial(muzerotrainer.make_muzero_networks, config=config)
        load_outputs = load_agent(
          env=env,
          config=config,
          builder=builder,
          network_factory=network_factory,
          seed_path=seed_path,
          use_latest=True,
          evaluation=True)
        reload(load_outputs.checkpointer, seed_path) # can use this to load in latest checkpoints
        action_dict = bwcfg.configurations['plan']['action_dict']

        print(f"\n----------------------- Evaluating lvl {lvl}")
        print(f"Muzero {groupname}")
        lvlsteps = [] # num steps for successful puzzles
        lvlepsr = [] # episode reward
        lvlsolved = [] # whether the puzzle is solved
        for irepeat in range(nrepeats):
          # print(f"irepeat {irepeat}")
          reload(load_outputs.checkpointer, seed_path)
          actor = load_outputs.actor
          timestep = env.reset()
          actor.observe_first(timestep)
          ends = False
          t = 0
          epsr = 0
          while not ends:
            action = actor.select_action(timestep.observation)
            timestep = env.step(action)
            steptype = timestep.step_type
            r = timestep.reward
            state = timestep.observation[0]
            t += 1 # increment time step
            epsr += r # episode cumulative reward
            if steptype==2: # if final timestep
              ends = True
              if (t<bwcfg.configurations['plan']['max_steps']-1) or (epsr>0.85):
                lvlsolved.append(1) # terminates early or solved at max step
                lvlsteps.append(t)
              else:
                lvlsolved.append(0)
            # print(f"\tt={t}, action_name: {action_dict[int(action)]}, r={round(float(r),5)}, ends={ends}")
          lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=[np.nan]
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        modelsolved.append(round(np.mean(lvlsolved),6))
        modelsolvedsem.append(round(sem(lvlsolved),6))
        modelreward.append(round(np.mean(lvlepsr),6))
        modelrewardsem.append(round(sem(lvlepsr),6))
        modelsteps.append(round(np.nanmean(lvlsteps), 6))
        modelstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

        print(f"Oracle")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of heuristic steps
        for irepeat in range(nrepeats):
          # print(f"irepeat {irepeat}")
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.oracle_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            # print(f"\tt={t}, action_name: {action_dict[int(a)]}, r={round(float(r),5)}")
          lvlepsr.append(epsr)
          lvlsolved.append(1)
          lvlsteps.append(len(expert_demo))
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        oraclesolved.append(round(np.mean(lvlsolved),6))
        oraclesolvedsem.append(round(sem(lvlsolved),6))
        oraclereward.append(round(np.mean(lvlepsr),6))
        oraclerewardsem.append(round(sem(lvlepsr),6))
        oraclesteps.append(round(np.mean(lvlsteps), 6))
        oraclestepssem.append(round(sem(lvlsteps), 6))

        print(f"Heuristic")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of expert steps
        for irepeat in range(nrepeats):
          # print(f"irepeat {irepeat}")
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.expert_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            # print(f"\tt={t}, action_name: {action_dict[int(a)]}, r={round(float(r),5)}")
          lvlepsr.append(epsr)
          lvlsolved.append(1)
          lvlsteps.append(len(expert_demo))
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        heuristicsolved.append(round(np.mean(lvlsolved),6))
        heuristicsolvedsem.append(round(sem(lvlsolved),6))
        heuristicreward.append(round(np.mean(lvlepsr),6))
        heuristicrewardsem.append(round(sem(lvlepsr),6))
        heuristicsteps.append(round(np.mean(lvlsteps), 6))
        heuristicstepssem.append(round(sem(lvlsteps), 6))

        print(f"Random")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of steps to solve
        for irepeat in range(nrepeats):
          state, _ = sim.reset()
          epsr = 0
          ends = False
          t=0
          while not ends:
            t+=1
            a = random.choice(range(sim.num_actions))
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            if terminated:
              ends = True
              lvlsolved.append(1)
              lvlsteps.append(t)
            elif truncated:
              ends=True
              lvlsolved.append(0)
          lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=[np.nan]
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        randomsolved.append(round(np.mean(lvlsolved),6))
        randomsolvedsem.append(round(sem(lvlsolved),6))
        randomreward.append(round(np.mean(lvlepsr),6))
        randomrewardsem.append(round(sem(lvlepsr),6))
        randomsteps.append(round(np.nanmean(lvlsteps), 6))
        randomstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

      print(f"modelsolved={modelsolved}\nmodelsolvedsem={modelsolvedsem}\
          \noraclesolved={oraclesolved}\noraclesolvedsem={oraclesolvedsem}\
          \nheuristicsolved={heuristicsolved}\nheuristicsolvedsem={heuristicsolvedsem}\
          \nrandomsolved={randomsolved}\nrandomsolvedsem={randomsolvedsem}\
          \nmodelsteps={modelsteps}\nmodelstepssem={modelstepssem}\
          \noraclesteps={oraclesteps}\noraclestepssem={oraclestepssem}\
          \nheuristicsteps={heuristicsteps}\nheuristicstepssem={heuristicstepssem}\
          \nrandomsteps={randomsteps}\nrandomstepssem={randomstepssem}")


    if eval_blocks and len(lvls)>0 and fixed_num_stacks!=None: # varying blocks, fix stacks
      modelsolved = [] # ratio of puzzles solved
      modelsolvedsem = [] 
      oraclesolved = []
      oraclesolvedsem = []
      heuristicsolved = []
      heuristicsolvedsem = []
      randomsolved = []
      randomsolvedsem = []
      modelreward = [] # avg episode reward
      modelrewardsem = []
      oraclereward = []
      oraclerewardsem = []
      heuristicreward = []
      heuristicrewardsem = []
      randomreward = []
      randomrewardsem = []
      modelsteps = [] # num of steps for solving a puzzle 
      modelstepssem = []
      oraclesteps = []
      oraclestepssem = []
      heuristicsteps = [] # num of steps for solving a puzzle
      heuristicstepssem = []
      randomsteps = [] # num of steps for solving a puzzle
      randomstepssem = []
      for lvl in lvls: # varying num blocks
        print(f"\n----------------------- Evaluating nblocks {lvl}, fixing nstacks={fixed_num_stacks}")
        print(f"Muzero {groupname}")
        lvlsteps = [] # num steps for successful puzzles
        lvlepsr = [] # episode reward
        lvlsolved = [] # whether the puzzle is solved
        for irepeat in range(nrepeats):
          puzzle = [[], []]
          while len(puzzle[1])!=fixed_num_stacks: # goal nstacks match
            nb, input_stacks, goal_stacks = bwutils.sample_random_puzzle(puzzle_max_stacks=puzzle_max_stacks, 
                                                              puzzle_max_blocks=puzzle_max_blocks, 
                                                              stack_max_blocks=stack_max_blocks,
                                                              puzzle_num_blocks=lvl, 
                                                              curriculum=False, leak=False,
                                                              compositional=compositional, 
                                                              compositional_eval=compositional_eval, 
                                                              compositional_type=compositional_type, 
                                                              compositional_holdout=compositional_holdout,)
            puzzle = [input_stacks, goal_stacks]
          env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=lvl,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=puzzle,
                                        )
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
                evaluation=True,
                mcts_policy=mcts_policy,
                discretizer=discretizer,
            ),
            ActorCls=functools.partial(
              basics.BasicActor,
              observers=[muzerotrainer.MuObserver(period=100000)],
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
                reanalyze_ratio=0.25, 
                root_policy_coef=config.root_policy_coef,
                root_value_coef=config.root_value_coef,
                model_policy_coef=config.model_policy_coef,
                model_value_coef=config.model_value_coef,
                model_reward_coef=config.model_reward_coef,
            ))
          network_factory = functools.partial(muzerotrainer.make_muzero_networks, config=config)
          load_outputs = load_agent(
            env=env,
            config=config,
            builder=builder,
            network_factory=network_factory,
            seed_path=seed_path,
            use_latest=True,
            evaluation=True)
          reload(load_outputs.checkpointer, seed_path)
          action_dict = bwcfg.configurations['plan']['action_dict']

          actor = load_outputs.actor
          timestep = env.reset()
          actor.observe_first(timestep)
          ends = False
          t = 0
          epsr = 0
          while not ends:
            action = actor.select_action(timestep.observation)
            timestep = env.step(action)
            steptype = timestep.step_type
            r = timestep.reward
            state = timestep.observation[0]
            t += 1 # increment time step
            epsr += r # episode cumulative reward
            if steptype==2: # if final timestep
              ends = True
              if (t<bwcfg.configurations['plan']['max_steps']-1) or (epsr>0.85):
                lvlsolved.append(1) # terminates early or solved at max step
                lvlsteps.append(t)
              else:
                lvlsolved.append(0)
          lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=[np.nan]
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        modelsolved.append(round(np.mean(lvlsolved),6))
        modelsolvedsem.append(round(sem(lvlsolved),6))
        modelreward.append(round(np.mean(lvlepsr),6))
        modelrewardsem.append(round(sem(lvlepsr),6))
        modelsteps.append(round(np.nanmean(lvlsteps), 6))
        modelstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

        print(f"Oracle")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of heuristic steps
        for irepeat in range(nrepeats):
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.oracle_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
          lvlepsr.append(epsr)
          lvlsolved.append(1)
          lvlsteps.append(len(expert_demo))
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        oraclesolved.append(round(np.mean(lvlsolved),6))
        oraclesolvedsem.append(round(sem(lvlsolved),6))
        oraclereward.append(round(np.mean(lvlepsr),6))
        oraclerewardsem.append(round(sem(lvlepsr),6))
        oraclesteps.append(round(np.mean(lvlsteps), 6))
        oraclestepssem.append(round(sem(lvlsteps), 6))

        print(f"Heuristic")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of expert steps
        for irepeat in range(nrepeats):
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.expert_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
          lvlepsr.append(epsr)
          lvlsolved.append(1)
          lvlsteps.append(len(expert_demo))
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        heuristicsolved.append(round(np.mean(lvlsolved),6))
        heuristicsolvedsem.append(round(sem(lvlsolved),6))
        heuristicreward.append(round(np.mean(lvlepsr),6))
        heuristicrewardsem.append(round(sem(lvlepsr),6))
        heuristicsteps.append(round(np.mean(lvlsteps), 6))
        heuristicstepssem.append(round(sem(lvlsteps), 6))

        print(f"Random")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of steps to solve
        for irepeat in range(nrepeats):
          state, _ = sim.reset()
          epsr = 0
          ends = False
          t=0
          while not ends:
            t+=1
            a = random.choice(range(sim.num_actions))
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            if terminated:
              ends = True
              lvlsolved.append(1)
              lvlsteps.append(t)
            elif truncated:
              ends=True
              lvlsolved.append(0)
          lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=[np.nan]
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        randomsolved.append(round(np.mean(lvlsolved),6))
        randomsolvedsem.append(round(sem(lvlsolved),6))
        randomreward.append(round(np.mean(lvlepsr),6))
        randomrewardsem.append(round(sem(lvlepsr),6))
        randomsteps.append(round(np.nanmean(lvlsteps), 6))
        randomstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

      print(f"modelsolved={modelsolved}\nmodelsolvedsem={modelsolvedsem}\
          \noraclesolved={oraclesolved}\noraclesolvedsem={oraclesolvedsem}\
          \nheuristicsolved={heuristicsolved}\nheuristicsolvedsem={heuristicsolvedsem}\
          \nrandomsolved={randomsolved}\nrandomsolvedsem={randomsolvedsem}\
          \nmodelsteps={modelsteps}\nmodelstepssem={modelstepssem}\
          \noraclesteps={oraclesteps}\noraclestepssem={oraclestepssem}\
          \nheuristicsteps={heuristicsteps}\nheuristicstepssem={heuristicstepssem}\
          \nrandomsteps={randomsteps}\nrandomstepssem={randomstepssem}")


    if eval_test_puzzles:
      print(f"\n----------------------- Evaluating test_puzzles")
      print(f"Muzero {groupname}")
      from envs.blocksworld.test_puzzles import test_puzzles
      lvlsolved = []
      lvlepsr = []
      for puzzle in test_puzzles:
        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=None,
                                        compositional=False, compositional_eval=False,
                                        compositional_type=None, compositional_holdout=None,
                                        test_puzzle=puzzle,
                                        )
        load_outputs = load_agent(
                                env=env,
                                config=config,
                                builder=builder,
                                network_factory=network_factory,
                                seed_path=seed_path,
                                use_latest=True,
                                evaluation=True)
        reload(load_outputs.checkpointer, seed_path)
        actor = load_outputs.actor
        timestep = env.reset()
        actor.observe_first(timestep)
        ends = False
        t = 0
        epsr = 0
        while not ends:
          action = actor.select_action(timestep.observation)
          timestep = env.step(action)
          steptype = timestep.step_type
          r = timestep.reward
          state = timestep.observation[0]
          t += 1 # increment time step
          epsr += r # episode cumulative reward
          if steptype==2: # if final timestep
            ends = True
            if (t<bwcfg.configurations['plan']['max_steps']-1) or (epsr>0.85):
              lvlsolved.append(1) # terminates early or at max step
            else:
              lvlsolved.append(0)
        lvlepsr.append(epsr)
      print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
              \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})")

      print(f"Heuristic")
      lvlsolved = []
      lvlepsr = []
      for puzzle in test_puzzles:
        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=None,
                                        compositional=False, compositional_eval=False,
                                        compositional_type=None, compositional_holdout=None,
                                        test_puzzle=puzzle,
                                        )
        for irepeat in range(nrepeats):
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.expert_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
          lvlepsr.append(epsr)
          lvlsolved.append(1)
      print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
              \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})")

      print(f"Random")
      lvlsolved = []
      lvlepsr = []
      for puzzle in test_puzzles:
        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=None,
                                        compositional=False, compositional_eval=False,
                                        compositional_type=None, compositional_holdout=None,
                                        test_puzzle=puzzle,
                                        )
        for irepeat in range(nrepeats):
          state, _ = sim.reset()
          epsr = 0
          ends = False
          while not ends:
            a = random.choice(range(sim.num_actions))
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            if terminated:
              ends = True
              lvlsolved.append(1)
            elif truncated:
              ends=True
              lvlsolved.append(0)
          lvlepsr.append(epsr)
      print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
              \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})")

    if eval_num_stacks and fixed_num_blocks!=None:
      modelsolved = [] # ratio of puzzles solved
      modelsolvedsem = [] 
      heuristicsolved = []
      heuristicsolvedsem = []
      oraclesolved = []
      oraclesolvedsem = []
      randomsolved = []
      randomsolvedsem = []
      modelreward = [] # avg episode reward
      modelrewardsem = []
      oraclereward = []
      oraclerewardsem = []
      heuristicreward = []
      heuristicrewardsem = []
      randomreward = []
      randomrewardsem = []
      modelsteps = [] # num of steps for solving a puzzle 
      modelstepssem = []
      oraclesteps = []
      oraclestepssem = []
      heuristicsteps = [] # num of steps for solving a puzzle
      heuristicstepssem = []
      randomsteps = [] # num of steps for solving a puzzle
      randomstepssem = []

      for nstacks in range(2, fixed_num_blocks+1):
        puzzles = [] # puzzles for this num of stacks
        while len(puzzles)<nrepeats: 
          nb, input_stacks, goal_stacks = bwutils.sample_random_puzzle(puzzle_max_stacks=nstacks, 
                                                              puzzle_max_blocks=puzzle_max_blocks, 
                                                              stack_max_blocks=stack_max_blocks,
                                                              puzzle_num_blocks=fixed_num_blocks, 
                                                              curriculum=False, leak=False,
                                                              compositional=compositional, 
                                                              compositional_eval=compositional_eval, 
                                                              compositional_type=compositional_type, 
                                                              compositional_holdout=compositional_holdout,)
                                      
          assert nb == fixed_num_blocks
          if len(input_stacks)==nstacks or len(goal_stacks)==nstacks:
            puzzles.append([input_stacks, goal_stacks])

        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=fixed_num_blocks,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=puzzles[0],
                                        )
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
              evaluation=True,
              mcts_policy=mcts_policy,
              discretizer=discretizer,
          ),
          ActorCls=functools.partial(
            basics.BasicActor,
            observers=[muzerotrainer.MuObserver(period=100000)],
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
              reanalyze_ratio=0.25, 
              root_policy_coef=config.root_policy_coef,
              root_value_coef=config.root_value_coef,
              model_policy_coef=config.model_policy_coef,
              model_value_coef=config.model_value_coef,
              model_reward_coef=config.model_reward_coef,
          ))
        network_factory = functools.partial(muzerotrainer.make_muzero_networks, config=config)
        load_outputs = load_agent(
          env=env,
          config=config,
          builder=builder,
          network_factory=network_factory,
          seed_path=seed_path,
          use_latest=True,
          evaluation=True)
        reload(load_outputs.checkpointer, seed_path) # can use this to load in latest checkpoints
        action_dict = bwcfg.configurations['plan']['action_dict']

        print(f"\n----------------------- Evaluating on {nstacks} stacks ({fixed_num_blocks} blocks)")
        print(f"Muzero {groupname}")
        lvlsteps = [] # num steps for successful puzzles
        lvlepsr = [] # episode reward
        lvlsolved = [] # whether the puzzle is solved
        for irepeat, puzzle in zip(range(nrepeats), puzzles):
          # print(f"irepeat {irepeat}")
          env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=fixed_num_blocks,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=puzzle,
                                        )
          load_outputs = load_agent(
                                  env=env,
                                  config=config,
                                  builder=builder,
                                  network_factory=network_factory,
                                  seed_path=seed_path,
                                  use_latest=True,
                                  evaluation=True)
          reload(load_outputs.checkpointer, seed_path)
          actor = load_outputs.actor
          timestep = env.reset()
          actor.observe_first(timestep)
          ends = False
          t = 0
          epsr = 0
          while not ends:
            action = actor.select_action(timestep.observation)
            timestep = env.step(action)
            steptype = timestep.step_type
            r = timestep.reward
            state = timestep.observation[0]
            t += 1 # increment time step
            epsr += r # episode cumulative reward
            if steptype==2: # if final timestep
              ends = True
              if (t<bwcfg.configurations['plan']['max_steps']-1) or (epsr>0.85):
                lvlsolved.append(1) # terminates early or solved at max step
                lvlsteps.append(t)
              else:
                lvlsolved.append(0)
            # print(f"\tt={t}, action_name: {action_dict[int(action)]}, r={round(float(r),5)}, ends={ends}")
          lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=[np.nan]
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        modelsolved.append(round(np.mean(lvlsolved),6))
        modelsolvedsem.append(round(sem(lvlsolved),6))
        modelreward.append(round(np.mean(lvlepsr),6))
        modelrewardsem.append(round(sem(lvlepsr),6))
        modelsteps.append(round(np.nanmean(lvlsteps), 6))
        modelstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

        print(f"Oracle")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of heuristic steps
        for irepeat, puzzle in zip(range(nrepeats), puzzles):
          # print(f"irepeat {irepeat}")
          _, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=fixed_num_blocks,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=puzzle,
                                        )
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.oracle_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            # print(f"\tt={t}, action_name: {action_dict[int(a)]}, r={round(float(r),5)}")
          lvlepsr.append(epsr)
          lvlsolved.append(1)
          lvlsteps.append(len(expert_demo))
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        oraclesolved.append(round(np.mean(lvlsolved),6))
        oraclesolvedsem.append(round(sem(lvlsolved),6))
        oraclereward.append(round(np.mean(lvlepsr),6))
        oraclerewardsem.append(round(sem(lvlepsr),6))
        oraclesteps.append(round(np.mean(lvlsteps), 6))
        oraclestepssem.append(round(sem(lvlsteps), 6))

        print(f"Heuristic")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of expert steps
        for irepeat, puzzle in zip(range(nrepeats), puzzles):
          # print(f"irepeat {irepeat}")
          _, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=fixed_num_blocks,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=puzzle,
                                        )
          state, _ = sim.reset()
          epsr = 0
          expert_demo = bwutils.expert_demo_plan(sim)
          for t, a in enumerate(expert_demo):
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            # print(f"\tt={t}, action_name: {action_dict[int(a)]}, r={round(float(r),5)}")
          lvlepsr.append(epsr)
          lvlsolved.append(1)
          lvlsteps.append(len(expert_demo))
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        heuristicsolved.append(round(np.mean(lvlsolved),6))
        heuristicsolvedsem.append(round(sem(lvlsolved),6))
        heuristicreward.append(round(np.mean(lvlepsr),6))
        heuristicrewardsem.append(round(sem(lvlepsr),6))
        heuristicsteps.append(round(np.mean(lvlsteps), 6))
        heuristicstepssem.append(round(sem(lvlsteps), 6))

        print(f"Random")
        lvlsolved = []
        lvlepsr = []
        lvlsteps = [] # num of steps to solve
        for irepeat, puzzle in zip(range(nrepeats), puzzles):
          _, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=fixed_num_blocks,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=puzzle,
                                        )
          state, _ = sim.reset()
          epsr = 0
          ends = False
          t=0
          while not ends:
            t+=1
            a = random.choice(range(sim.num_actions))
            state, r, terminated, truncated, _ = sim.step(a)
            epsr += r # episode cumulative reward
            if terminated:
              ends = True
              lvlsolved.append(1)
              lvlsteps.append(t)
            elif truncated:
              ends=True
              lvlsolved.append(0)
          lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=[np.nan]
        print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
                \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})\
                \n\tavg steps {round(np.mean(lvlsteps), 6)} (sem={round(sem(lvlsteps), 6)})")
        randomsolved.append(round(np.mean(lvlsolved),6))
        randomsolvedsem.append(round(sem(lvlsolved),6))
        randomreward.append(round(np.mean(lvlepsr),6))
        randomrewardsem.append(round(sem(lvlepsr),6))
        randomsteps.append(round(np.nanmean(lvlsteps), 6))
        randomstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

      print(f"modelsolved={modelsolved}\nmodelsolvedsem={modelsolvedsem}\
          \noraclesolved={oraclesolved}\noraclesolvedsem={oraclesolvedsem}\
          \nheuristicsolved={heuristicsolved}\nheuristicsolvedsem={heuristicsolvedsem}\
          \nrandomsolved={randomsolved}\nrandomsolvedsem={randomsolvedsem}\
          \nmodelsteps={modelsteps}\nmodelstepssem={modelstepssem}\
          \noraclesteps={oraclesteps}\noraclestepssem={oraclestepssem}\
          \nheuristicsteps={heuristicsteps}\nheuristicstepssem={heuristicstepssem}\
          \nrandomsteps={randomsteps}\nrandomstepssem={randomstepssem}")


    if eval_steps and lvls!=None:
      modelsolved = [np.nan]*max_oracle_steps # ratio of puzzles solved
      modelsolvedsem = [np.nan]*max_oracle_steps
      oraclesolved = [np.nan]*max_oracle_steps
      oraclesolvedsem = [np.nan]*max_oracle_steps
      heuristicsolved = [np.nan]*max_oracle_steps
      heuristicsolvedsem = [np.nan]*max_oracle_steps
      randomsolved = [np.nan]*max_oracle_steps
      randomsolvedsem = [np.nan]*max_oracle_steps
      modelreward = [np.nan]*max_oracle_steps # avg episode reward
      modelrewardsem = [np.nan]*max_oracle_steps
      oraclereward = [np.nan]*max_oracle_steps
      oraclerewardsem = [np.nan]*max_oracle_steps
      heuristicreward = [np.nan]*max_oracle_steps
      heuristicrewardsem = [np.nan]*max_oracle_steps
      randomreward = [np.nan]*max_oracle_steps
      randomrewardsem = [np.nan]*max_oracle_steps
      modelsteps = [np.nan]*max_oracle_steps # num of steps for solving a puzzle 
      modelstepssem = [np.nan]*max_oracle_steps
      oraclesteps = [np.nan]*max_oracle_steps
      oraclestepssem = [np.nan]*max_oracle_steps
      heuristicsteps = [np.nan]*max_oracle_steps # num of steps for solving a puzzle
      heuristicstepssem = [np.nan]*max_oracle_steps
      randomsteps = [np.nan]*max_oracle_steps # num of steps for solving a puzzle
      randomstepssem = [np.nan]*max_oracle_steps
      for irepeat in range(nsamples):
        # sample a random puzzle
        nb, input_stacks, goal_stacks = bwutils.sample_random_puzzle(puzzle_max_stacks=puzzle_max_stacks, 
                                                              puzzle_max_blocks=puzzle_max_blocks, 
                                                              stack_max_blocks=stack_max_blocks,
                                                              puzzle_num_blocks=random.choice(lvls), 
                                                              curriculum=False, leak=False,
                                                              compositional=compositional, 
                                                              compositional_eval=compositional_eval, 
                                                              compositional_type=compositional_type, 
                                                              compositional_holdout=compositional_holdout,)
        test_puzzle = [input_stacks, goal_stacks]
        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        evaluation=True,
                                        eval_puzzle_num_blocks=None,
                                        compositional=compositional, compositional_eval=compositional_eval,
                                        compositional_type=compositional_type, compositional_holdout=compositional_holdout,
                                        test_puzzle=test_puzzle,
                                        )
        oracle_demo = bwutils.oracle_demo_plan(sim)
        oraclensteps = len(oracle_demo)
        print(f"oraclensteps {oraclensteps}")

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
              evaluation=True,
              mcts_policy=mcts_policy,
              discretizer=discretizer,
          ),
          ActorCls=functools.partial(
            basics.BasicActor,
            observers=[muzerotrainer.MuObserver(period=100000)],
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
              reanalyze_ratio=0.25, 
              root_policy_coef=config.root_policy_coef,
              root_value_coef=config.root_value_coef,
              model_policy_coef=config.model_policy_coef,
              model_value_coef=config.model_value_coef,
              model_reward_coef=config.model_reward_coef,
          ))
        network_factory = functools.partial(muzerotrainer.make_muzero_networks, config=config)
        load_outputs = load_agent(
          env=env,
          config=config,
          builder=builder,
          network_factory=network_factory,
          seed_path=seed_path,
          use_latest=True,
          evaluation=True)
        action_dict = bwcfg.configurations['plan']['action_dict']

        # print(f"Muzero {groupname}")
        lvlsolved = [] if np.isnan(modelsolved[oraclensteps]).all() else modelsolved[oraclensteps]
        lvlsteps = [] if np.isnan(modelsteps[oraclensteps]).all() else modelsteps[oraclensteps]
        lvlepsr = [] if np.isnan(modelreward[oraclensteps]).all() else modelreward[oraclensteps]
        reload(load_outputs.checkpointer, seed_path)
        actor = load_outputs.actor
        timestep = env.reset()
        actor.observe_first(timestep)
        ends = False
        t = 0
        epsr = 0
        while not ends:
          action = actor.select_action(timestep.observation)
          timestep = env.step(action)
          steptype = timestep.step_type
          r = timestep.reward
          state = timestep.observation[0]
          t += 1 # increment time step
          epsr += r # episode cumulative reward
          if steptype==2: # if final timestep
            ends = True
            if (t<bwcfg.configurations['plan']['max_steps']-1) or (epsr>0.85):
              lvlsolved.append(1) # terminates early or solved at max step
              lvlsteps.append(t)
            else:
              lvlsolved.append(0)
        lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=np.nan
        modelsolved[oraclensteps] = lvlsolved
        modelreward[oraclensteps] = lvlepsr
        modelsteps[oraclensteps] = lvlsteps

        # print(f"Oracle")
        lvlsolved = [] if np.isnan(oraclesolved[oraclensteps]).all() else oraclesolved[oraclensteps]
        lvlsteps = [] if np.isnan(oraclesteps[oraclensteps]).all() else oraclesteps[oraclensteps]
        lvlepsr = [] if np.isnan(oraclereward[oraclensteps]).all() else oraclereward[oraclensteps]
        state, _ = sim.reset()
        epsr = 0
        expert_demo = bwutils.oracle_demo_plan(sim)
        for t, a in enumerate(expert_demo):
          state, r, terminated, truncated, _ = sim.step(a)
          epsr += r # episode cumulative reward
        lvlepsr.append(epsr)
        lvlsolved.append(1)
        lvlsteps.append(len(expert_demo))
        oraclesolved[oraclensteps] = lvlsolved
        oraclereward[oraclensteps] = lvlepsr
        oraclesteps[oraclensteps] = lvlsteps

        # print(f"Heuristic")
        lvlsolved = [] if np.isnan(heuristicsolved[oraclensteps]).all() else heuristicsolved[oraclensteps]
        lvlsteps = [] if np.isnan(heuristicsteps[oraclensteps]).all() else heuristicsteps[oraclensteps]
        lvlepsr = [] if np.isnan(heuristicreward[oraclensteps]).all() else heuristicreward[oraclensteps]
        state, _ = sim.reset()
        epsr = 0
        expert_demo = bwutils.expert_demo_plan(sim)
        for t, a in enumerate(expert_demo):
          state, r, terminated, truncated, _ = sim.step(a)
          epsr += r # episode cumulative reward
        lvlepsr.append(epsr)
        lvlsolved.append(1)
        lvlsteps.append(len(expert_demo))
        heuristicsolved[oraclensteps] = lvlsolved
        heuristicreward[oraclensteps] = lvlepsr
        heuristicsteps[oraclensteps] = lvlsteps

        # print(f"Random")
        lvlsolved = [] if np.isnan(randomsolved[oraclensteps]).all() else randomsolved[oraclensteps]
        lvlsteps = [] if np.isnan(randomsteps[oraclensteps]).all() else randomsteps[oraclensteps]
        lvlepsr = [] if np.isnan(randomreward[oraclensteps]).all() else randomreward[oraclensteps]
        state, _ = sim.reset()
        epsr = 0
        ends = False
        t=0
        while not ends:
          t+=1
          a = random.choice(range(sim.num_actions))
          state, r, terminated, truncated, _ = sim.step(a)
          epsr += r # episode cumulative reward
          if terminated:
            ends = True
            lvlsolved.append(1)
            lvlsteps.append(t)
          elif truncated:
            ends=True
            lvlsolved.append(0)
        lvlepsr.append(epsr)
        if len(lvlsteps)==0:
          lvlsteps=np.nan
        randomsolved[oraclensteps] = lvlsolved
        randomreward[oraclensteps] = lvlepsr
        randomsteps[oraclensteps] = lvlsteps

      for oraclensteps in range(max_oracle_steps):
        modelsolvedsem[oraclensteps] = round(sem(modelsolved[oraclensteps]),6)
        modelsolved[oraclensteps] = round(np.mean(modelsolved[oraclensteps]),6)
        modelrewardsem[oraclensteps] = round(sem(modelreward[oraclensteps]),6)
        modelreward[oraclensteps] = round(np.mean(modelreward[oraclensteps]),6)
        modelstepssem[oraclensteps] = round(sem(modelsteps[oraclensteps], nan_policy="omit"), 6)
        modelsteps[oraclensteps] = round(np.nanmean(modelsteps[oraclensteps]), 6)

        oraclesolvedsem[oraclensteps] = round(sem(oraclesolved[oraclensteps]),6)
        oraclesolved[oraclensteps] = round(np.mean(oraclesolved[oraclensteps]),6)
        oraclerewardsem[oraclensteps] = round(sem(oraclereward[oraclensteps]),6)
        oraclereward[oraclensteps] = round(np.mean(oraclereward[oraclensteps]),6)
        oraclestepssem[oraclensteps] = round(sem(oraclesteps[oraclensteps]), 6)
        oraclesteps[oraclensteps] = round(np.mean(oraclesteps[oraclensteps]), 6)
        
        heuristicsolvedsem[oraclensteps] = round(sem(heuristicsolved[oraclensteps]),6)
        heuristicsolved[oraclensteps] = round(np.mean(heuristicsolved[oraclensteps]),6)
        heuristicrewardsem[oraclensteps] = round(sem(heuristicreward[oraclensteps]),6)
        heuristicreward[oraclensteps] = round(np.mean(heuristicreward[oraclensteps]),6)
        heuristicstepssem[oraclensteps] = round(sem(heuristicsteps[oraclensteps]), 6)
        heuristicsteps[oraclensteps] = round(np.mean(heuristicsteps[oraclensteps]), 6)

        randomsolvedsem[oraclensteps] = round(sem(randomsolved[oraclensteps]),6)
        randomsolved[oraclensteps] = round(np.mean(randomsolved[oraclensteps]),6)
        randomrewardsem[oraclensteps] = round(sem(randomreward[oraclensteps]),6)
        randomreward[oraclensteps] = round(np.mean(randomreward[oraclensteps]),6)
        randomstepssem[oraclensteps] = round(sem(randomsteps[oraclensteps], nan_policy="omit"), 6)
        randomsteps[oraclensteps] = round(np.nanmean(randomsteps[oraclensteps]), 6)

      print(f"modelsolved={modelsolved}\nmodelsolvedsem={modelsolvedsem}\
          \nrandomsolved={randomsolved}\nrandomsolvedsem={randomsolvedsem}\
          \nmodelsteps={modelsteps}\nmodelstepssem={modelstepssem}\
          \noraclesteps={oraclesteps}\noraclestepssem={oraclestepssem}\
          \nheuristicsteps={heuristicsteps}\nheuristicstepssem={heuristicstepssem}\
          \nrandomsteps={randomsteps}\nrandomstepssem={randomstepssem}")


'''
salloc -p gpu_test -t 0-03:00 --mem=80000 --gres=gpu:1

salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

python library/eval_muzero.py 
'''

if __name__ == "__main__":
  random.seed(0)
  main(
        eval_lvls=False, lvls=[2,3,4,5,6], # whether to eval on varying num blocks, nblocks also varied
        eval_blocks=True, fixed_num_stacks=2, # whether to eval on varying num blocks while fixing num stacks
        puzzle_max_stacks=5, # model config
        puzzle_max_blocks=10, # model config
        stack_max_blocks=7, # model config
        sparse_reward=False, # whether the training is sparse reward
        compositional=False, # whether the training setting is compositional
        compositional_eval=False, compositional_type='newblock', compositional_holdout=[2,3,5,7], # whether to eval on comp holdout
        eval_test_puzzles=False, # whether to eval on 100 jBrain puzzles
        eval_num_stacks=False, fixed_num_blocks=4, # whether to eval on varying num stacks while fixing num blocks
        eval_steps=False, max_oracle_steps=40, nsamples=500, # whether to eval on solution lengths
        nrepeats=50, # num samples for all analyses except eval_steps
        groupname='M4~10-2v8max5-10-7', # model to load
      )

'''
'Msparse4~10-2v8max5-10-7'
  stack_max_blocks=7, 
  puzzle_max_blocks=10,
  puzzle_max_stacks=5,
  sparse_reward=True,
  curriculum: 4~10 (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=False,

'M4~10-2v8max5-10-7'
  stack_max_blocks=7, 
  puzzle_max_blocks=10,
  puzzle_max_stacks=5,
  sparse_reward=False,
  curriculum: 4~10 (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=False,

'Msparse4~10comp5v8max2-11-7'
  stack_max_blocks=7, 
  puzzle_max_blocks=11,
  puzzle_max_stacks=2,
  sparse_reward=True,
  curriculum: 4~10 (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=True, compositional_type='newblock', compositional_holdout=[2,3,5,7], 

'Mu4~10comp5v8max2-11-7'
  stack_max_blocks=7, 
  puzzle_max_blocks=11,
  puzzle_max_stacks=2,
  sparse_reward=False,
  curriculum: 4~10 (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=True, compositional_type='newblock', compositional_holdout=[2,3,5,7], 

'Msparse4~10comp5v8max1-11-7'
  stack_max_blocks=7, 
  puzzle_max_blocks=11,
  puzzle_max_stacks=1,
  sparse_reward=True,
  curriculum: 4~10 (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=True, compositional_type='newblock', compositional_holdout=[2,3,5,7],

'Mu4~10comp5v8max1-11-7'
  stack_max_blocks=7, 
  puzzle_max_blocks=11,
  puzzle_max_stacks=1,
  sparse_reward=False,
  curriculum: 4~10 (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=True, compositional_type='newblock', compositional_holdout=[2,3,5,7],

'muz4+8v-2'
  stack_max_blocks=7, 
  puzzle_max_blocks=10,
  puzzle_max_stacks=5,
  sparse_reward=False,
  curriculum: 4+ (no leak),
  up_threshold: 0.8,
  down_threshold: -2,
  compositional=False,

'muz2~10long5v8'
  stack_max_blocks=7, 
  puzzle_max_blocks=10,
  puzzle_max_stacks=5,
  sparse_reward=False,
  curriculum: 2~10 (no leak),
  up_threshold: 0.8,
  down_threshold: 0.5,
  compositional=False,

'''