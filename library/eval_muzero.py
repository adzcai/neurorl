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


def make_test_environment(puzzle_num_blocks: int, 
                        compositional, compositional_type, compositional_holdout):
  """
  Make plan env for testing
  Returns:
    dm_env.Environment object, with multiple elements wrapped together (simulator, observation, action, reward, single precision).
  """
  # create dm_env
  sim = plan.Simulator(evaluation=True, 
                      eval_puzzle_num_blocks=puzzle_num_blocks,
                      compositional=compositional, compositional_type=compositional_type, compositional_holdout=compositional_holdout)
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


def main(lvls, compositional, compositional_type, compositional_holdout,
        groupname, searchname='initial', 
        nrepeats=100):
    # EXAMPLE: CHANGE AS NEEDED
    default_log_dir = os.environ['RL_RESULTS_DIR']
    base_dir = os.path.join(
      default_log_dir,
      searchname,  # search name
      groupname  # group name
    )
    seed_path, env_kwargs, agent_kwargs = load_settings(      
      base_dir=base_dir,
      run='.')

    from configs.plan_trainer_muzero import make_muzero_networks, MuObserver
    import envs.blocksworld.cfg as bwcfg
    from td_agents import muzero
    import functools
    import mctx
    import library.utils as utils
    import envs.blocksworld.utils as bwutils
    config = muzero.Config(**agent_kwargs)
    action_dict = bwcfg.configurations['plan']['action_dict']
    # print(f"config {config}")

    for lvl in lvls:
      env, sim = make_test_environment(puzzle_num_blocks=lvl, 
                              compositional=compositional, compositional_type=compositional_type, compositional_holdout=compositional_holdout)
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
          observers=[MuObserver(period=100000)],
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
      network_factory = functools.partial(make_muzero_networks, config=config)
      load_outputs = load_agent(
        env=env,
        config=config,
        builder=builder,
        network_factory=network_factory,
        seed_path=seed_path,
        use_latest=True,
        evaluation=True)
      reload(load_outputs.checkpointer, seed_path) # can use this to load in latest checkpoints

      print(f"\n----------------------- Evaluating lvl {lvl}")
      print(f"Muzero {groupname}")
      lvlsolved = []
      lvlepsr = []
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
              lvlsolved.append(1) # terminates early or at max step
            else:
              lvlsolved.append(0)
          # print(f"\tt={t}, action_name: {action_dict[int(action)]}, r={round(float(r),5)}, ends={ends}")
        lvlepsr.append(epsr)
      print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
              \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})")

      print(f"Expert")
      lvlsolved = []
      lvlepsr = []
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


    print(f"\n----------------------- Evaluating test_puzzles")
    print(f"Muzero {groupname}")
    from envs.blocksworld.test_puzzles import test_puzzles
    lvlsolved = []
    lvlepsr = []
    for puzzle in test_puzzles:
      env, sim = make_test_environment(puzzle_num_blocks=None, 
                                      compositional=False, 
                                        compositional_type=None, compositional_holdout=None,
                                      test_puzzle=puzzle)
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
        # print(f"\tt={t}, action_name: {action_dict[int(action)]}, r={round(float(r),5)}, ends={ends}")
      lvlepsr.append(epsr)
    print(f"\tavg solved: {round(np.mean(lvlsolved),6)} (sem={round(sem(lvlsolved),6)})\
            \n\tavg epsr: {round(np.mean(lvlepsr),6)} (sem={round(sem(lvlepsr),6)})")

    print(f"Expert")
    lvlsolved = []
    lvlepsr = []
    for puzzle in test_puzzles:
      env, sim = make_test_environment(puzzle_num_blocks=None, 
                                      compositional=False, 
                                        compositional_type=None, compositional_holdout=None,
                                      test_puzzle=puzzle)
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
      env, sim = make_test_environment(puzzle_num_blocks=None, 
                                      compositional=False, 
                                        compositional_type=None, compositional_holdout=None,
                                      test_puzzle=puzzle)
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




'''
salloc -p test -t 0-01:00 --mem=200000 

salloc -p gpu_test -t 0-01:00 --mem=8000 --gres=gpu:1
module load python/3.10.12-fasrc01
mamba activate neurorl
'''

if __name__ == "__main__":
  random.seed(0)
  main(lvls=[2,3,4], 
        compositional=False, 
            compositional_type='newblock', compositional_holdout=[2,3,5,7],
        groupname='muz2~10long5v8', 
        nrepeats=200)

'''
  puzzle_max_blocks: 10
  puzzle_max_stacks: 5
  stack_max_blocks: 7
  curriculum: 2 ~ 4 (no leak)
  compositional: False
'''