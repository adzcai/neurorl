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
import numpy as np
from glob import glob
import pickle

from td_agents import basics

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
  # first load configs
  if seed_path is None:
    assert base_dir is not None and run is not None, 'set values for finding path'
    seed_path = glob(os.path.join(base_dir, run, '*'))[0]
  print(f"seed_path {seed_path}")
  config_file = os.path.join(seed_path, 'config.pkl')
  config = load_config(config_file)
  print(f"config: {config}")
  # final_agent_config = config['final_agent_config']
  # final_env_config = config['final_env_config']
  final_agent_config = config
  final_env_config = {}
  return seed_path, final_env_config, final_agent_config

from acme import wrappers as acme_wrappers
import dm_env
from envs.blocksworld import plan
def make_test_environment(puzzle_num_blocks: int=2, **kwargs) -> dm_env.Environment:
  """
  Make plan env for testing
  Returns:
    dm_env.Environment object, with multiple elements wrapped together (simulator, observation, action, reward, single precision).
  """
  print(f"\n-------------Evaluating level {puzzle_num_blocks}")
  # create dm_env
  sim = plan.Simulator(evaluation=puzzle_num_blocks)
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
  return acme_wrappers.wrap_all(env, wrapper_list)

def main():
    # EXAMPLE: CHANGE AS NEEDED
    default_log_dir = os.environ['RL_RESULTS_DIR']
    base_dir = os.path.join(
      default_log_dir,
      'initial',  # search name
      'easy2-10plan'  # group name
    )
    print(f"base_dir {base_dir}")

    seed_path, env_kwargs, agent_kwargs = load_settings(
      # seed_path = "/n/home04/yichenli/rl_results/initial/runs-2024.04.19-16.57/",
      base_dir=base_dir,
      run='.')

    from configs.plan_trainer import make_qlearning_networks, QObserver, make_environment
    import envs.blocksworld.cfg as bwcfg
    from td_agents import q_learning, basics
    import functools

    config = q_learning.Config(**agent_kwargs)
    env = make_test_environment(puzzle_num_blocks=3, *env_kwargs)

    builder = basics.Builder(
      config=config,
      ActorCls=functools.partial(
        basics.BasicActor,
        observers=[QObserver(period=50000)],
        ),
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        evaluation=True,
      ),
      LossFn=q_learning.R2D2LossFn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))

    # network_factory = functools.partial(q_learning.make_minigrid_networks, config=config)
    network_factory = functools.partial(make_qlearning_networks, config=config)

    load_outputs = load_agent(
      env=env,
      config=config,
      builder=builder,
      network_factory=network_factory,
      seed_path=seed_path,
      use_latest=True,
      evaluation=True)

    # can use this to load in latest checkpoints
    reload(load_outputs.checkpointer, seed_path)

    action_dict = bwcfg.configurations['plan']['action_dict']
    print(f"\n-----------------------Evaluating")
    lvlsolved = []
    lvlepsr = []
    for irepeat in range(100):
      print(f"irepeat {irepeat}")
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
        if steptype==2: # final timestep
          ends = True
          if t<bwcfg.configurations['plan']['max_steps']-1:
            lvlsolved.append(1)
          else:
            lvlsolved.append(0)
        print(f"\tt={t}, action_name: {action_dict[int(action)]}, r={round(float(r),5)}, ends={ends}")
        t += 1
        epsr += r
      lvlepsr.append(epsr)


    import numpy as np
    print(f"\n\nlvlsolved {lvlsolved}\n\tavg {np.mean(lvlsolved)}\nlvlepsr {lvlepsr}\n\tavg {np.mean(lvlepsr)}")
        

if __name__ == "__main__":
    main()