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

import configs.blocksworld.train_plan_qlearning as qltrainer
import envs.blocksworld.cfg as bwcfg
from td_agents import q_learning
import functools
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
  # get all directories from year
  # load checkpoint
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
  if seed_path is None: # seed_path = "/n/home04/yichenli/rl_results/initial/runs-2024.04.19-16.57/"
    assert base_dir is not None and run is not None, 'set values for finding path'
    seed_path = glob(os.path.join(base_dir, run, '*'))[0]
  config_file = os.path.join(seed_path, 'config.pkl')
  config = load_config(config_file)
  final_agent_config = config
  final_env_config = {}
  return seed_path, final_env_config, final_agent_config



def make_test_environment( 
                        puzzle_max_blocks, 
                        stack_max_blocks, 
                        sparse_reward,
                        ):
  """
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
        puzzle_max_stacks, 
        puzzle_max_blocks, 
        stack_max_blocks, 
        sparse_reward,
        groupname, searchname='initial', 
        nrepeats=100,
        ):
    '''
      lvls: list[int]
        list of puzzle_num_blocks to evaluate, should be within range [2, puzzle_max_blocks]
      groupname: str
      searchname: str
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


    config = q_learning.Config(**agent_kwargs)



  def tmp_obs_encoder(
    inputs: acme_wrappers.observation_action_reward.OAR,
    num_actions: int,
    stack_max_blocks: int=stack_max_blocks,
	puzzle_max_blocks: int=puzzle_max_blocks,
	puzzle_max_stacks: int=puzzle_max_stacks):
    num_areas = bwcfg.configurations['parse']['num_areas']
    num_fibers = bwcfg.configurations['parse']['num_fibers']
    assert type(bwcfg.configurations['parse']['num_areas'])==int, f"{bwcfg.configurations['parse']['num_areas']}"
    curstack_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
    cut1 = stack_max_blocks
    goalstack_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
    cut2 = cut1 + stack_max_blocks
    fiber_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
    cut3 = cut2 + num_fibers
    lastass_embed = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())
    cut4 = cut3 + num_areas
    nblass_embed = hk.Linear(512, w_init=hk.initializers.TruncatedNormal())
    cut5 = cut4 + num_areas
    nass_embed = hk.Linear(512, w_init=hk.initializers.TruncatedNormal())
    cut6 = cut5 + num_areas
    topn_embed = hk.Linear(64, w_init=hk.initializers.TruncatedNormal())
    cut7 = cut6+1
    topa_embed = hk.Linear(64, w_init=hk.initializers.TruncatedNormal())
    cut8 = cut7+1
    topb_embed = hk.Linear(64, w_init=hk.initializers.TruncatedNormal())
    cut9 = cut8+1
    islast_embed = hk.Linear(64, w_init=hk.initializers.TruncatedNormal())
    reward_embed = hk.Linear(128, w_init=hk.initializers.RandomNormal())
    action_embed = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())
    mlp = hk.nets.MLP([512,512,512,512], activate_final=True) # default RELU activations between layers (and after final layer)
    def fn(x):
      x = jnp.concatenate((
      curstack_embed(jax.nn.one_hot(x.observation[:cut1], puzzle_max_blocks).reshape(-1)),
      goalstack_embed(jax.nn.one_hot(x.observation[cut1:cut2], puzzle_max_blocks).reshape(-1)),
      fiber_embed(jax.nn.one_hot(x.observation[cut2:cut3], 2).reshape(-1)),
      lastass_embed(jax.nn.one_hot(x.observation[cut3:cut4], max_assemblies).reshape(-1)),
      nblass_embed(jax.nn.one_hot(x.observation[cut4:cut5], max_assemblies).reshape(-1)),
      nass_embed(jax.nn.one_hot(x.observation[cut5:cut6], max_assemblies).reshape(-1)),
      topn_embed(jax.nn.one_hot(x.observation[cut6:cut7], 3).reshape(-1)),
      topa_embed(jax.nn.one_hot(x.observation[cut7:cut8], max_assemblies).reshape(-1)),
      topb_embed(jax.nn.one_hot(x.observation[cut8:cut9], puzzle_max_blocks).reshape(-1)),
      islast_embed(jax.nn.one_hot(x.observation[cut9:], 2).reshape(-1)),
      reward_embed(jnp.expand_dims(x.reward, 0)), 
      action_embed(jax.nn.one_hot(x.action, num_actions))  
      ))
      x = jax.nn.relu(x)
      x = mlp(x)
      return x
    has_batch_dim = inputs.reward.ndim > 0
    if has_batch_dim: # have batch dimension
      fn = jax.vmap(fn)
    return fn(inputs)


    qltrainer.observation_encoder = lambda inputs, num_actions: tmp_obs_encoder(inputs=inputs, num_actions=num_actions)


    if eval_lvls and len(lvls)>0:
      qlsolved = [] # ratio of puzzles solved
      qlsolvedsem = [] 
      oraclesolved = []
      oraclesolvedsem = []
      randomsolved = []
      randomsolvedsem = []
      qlreward = [] # avg episode reward
      qlrewardsem = []
      oraclereward = []
      oraclerewardsem = []
      randomreward = []
      randomrewardsem = []
      qlsteps = [] # num of steps for solving a puzzle 
      qlstepssem = []
      oraclesteps = []
      oraclestepssem = []
      randomsteps = [] # num of steps for solving a puzzle
      randomstepssem = []
      for lvl in lvls:
        env, sim = make_test_environment(
                                        puzzle_max_stacks=puzzle_max_stacks,
                                        puzzle_max_blocks=puzzle_max_blocks,
                                        stack_max_blocks=stack_max_blocks,
                                        sparse_reward=sparse_reward,
                                        )
        action_dict = bwcfg.configurations['parse']['action_dict']
        builder = basics.Builder(
          config=config,
          ActorCls=functools.partial(
            basics.BasicActor,observers=[qltrainer.QObserver(period=1000000)],),
          get_actor_core_fn=functools.partial(basics.get_actor_core,evaluation=True,),
          LossFn=q_learning.R2D2LossFn(discount=config.discount,
            importance_sampling_exponent=config.importance_sampling_exponent,
            burn_in_length=config.burn_in_length,max_replay_size=config.max_replay_size,
            max_priority_weight=config.max_priority_weight,bootstrap_n=config.bootstrap_n,))
        network_factory = functools.partial(qltrainer.make_qlearning_networks, config=config)
        load_outputs = load_agent(env=env,config=config,builder=builder,network_factory=network_factory,
								seed_path=seed_path,use_latest=True,evaluation=True)
        reload(load_outputs.checkpointer, seed_path)

        print(f"\n----------------------- Evaluating lvl {lvl}")
        print(f"Qlearning {groupname}")
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
        qlsolved.append(round(np.mean(lvlsolved),6))
        qlsolvedsem.append(round(sem(lvlsolved),6))
        qlreward.append(round(np.mean(lvlepsr),6))
        qlrewardsem.append(round(sem(lvlepsr),6))
        qlsteps.append(round(np.nanmean(lvlsteps), 6))
        qlstepssem.append(round(sem(lvlsteps, nan_policy="omit"), 6))

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

      print(f"qlsolved={qlsolved}\nqlsolvedsem={qlsolvedsem}\
          \noraclesolved={oraclesolved}\noraclesolvedsem={oraclesolvedsem}\
          \nrandomsolved={randomsolved}\nrandomsolvedsem={randomsolvedsem}\
          \nqlsteps={qlsteps}\nqlstepssem={qlstepssem}\
          \noraclesteps={oraclesteps}\noraclestepssem={oraclestepssem}\
          \nrandomsteps={randomsteps}\nrandomstepssem={randomstepssem}")





'''
salloc -p gpu_test -t 0-03:00 --mem=80000 --gres=gpu:1

salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

python configs/blocksworld/eval_parse_qlearning.py
'''

if __name__ == "__main__":
  random.seed(0)
  main(
        eval_lvls=True, lvls=[2,3,4,5], # whether to eval on varying num blocks (num stacks can vary)
        puzzle_max_stacks=5, # model config
        puzzle_max_blocks=10, # model config
        stack_max_blocks=7, # model config
        sparse_reward=True, # whether the training is sparse reward
        nrepeats=200, # num samples for all analyses except eval_steps
        groupname='Qsparse5onlycomp2perc25max5-10-7', # which model to load
      )
