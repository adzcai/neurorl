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


def load_settings(
  base_dir: str = None,
  run: str = None,
  seed_path: str = None):
  # first load configs
  if seed_path is None:
    assert base_dir is not None and run is not None, 'set values for finding path'
    seed_path = glob(os.path.join(base_dir, run, '*'))[0]
  env_file = os.path.join(seed_path, 'env_config_kw.pkl')
  env_kwargs = load_config(env_file)

  config_file = os.path.join(seed_path, 'config.pkl')
  config_kwargs = load_config(config_file)
  return seed_path, env_kwargs, config_kwargs


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
  print('loading', ckpt_path)
  status = checkpointer._checkpoint.restore(ckpt_path)

if __name__ == "__main__":
    # EXAMPLE: CHANGE AS NEEDED
    default_log_dir = os.environ['RL_RESULTS_DIR']
    base_dir = os.path.join(
      default_log_dir,
      'usfa_dyna2',
      'usfa_dyna-60-ind')

    seed_path, env_kwargs, config_kwargs = load_settings(
      base_dir=base_dir,
      run='agen=object_usfa_dyna,task=False,task=False,weig=1.0,sep_=True,disc=0.99,sf_l=[128],unwe=0.0,tile=12,back=False,num_=2,trai=1,test=False,eval=True,term=True')

    from configs.minigrid_trainer import make_environment
    from td_agents import q_learning
    from td_agents import basics
    import functools

    config = q_learning.Config(**env_kwargs)
    env = make_environment(*env_kwargs)

    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        linear_epsilon=config.linear_epsilon,
      ),
      LossFn=q_learning.R2D2LossFn(
        discount=config.discount,
        
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))

    network_factory = functools.partial(
            q_learning.make_minigrid_networks,
            config=config)

    load_agent(
      env=env,
      config=config,
      builder=builder,
      network_factory=network_factory,
      seed_path=seed_path,
      use_latest=True,
      evaluation=True)
    import ipdb; ipdb.set_trace()