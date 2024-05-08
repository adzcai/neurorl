from typing import NamedTuple, Any, Callable
import os
import logging

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'# https://github.com/google/jax/issues/8302
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
import pprint
from scipy.stats import sem
from acme import wrappers as acme_wrappers
import dm_env
from envs.language import langenv

import configs.language.train_muzero as muzerotrainer 
import envs.language.cfg as langcfg
from td_agents import muzero
import functools
import mctx
import library.utils as utils
import envs.language.utils as langutils

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
						evaluation, 
						eval_sentence_complexity,
						spacing,
						compositional, 
						compositional_eval,
						compositional_holdout,
						test_sentence,
						):
	sim = langenv.Simulator(
						spacing=spacing,
						evaluation=evaluation,
						eval_sentence_complexity=eval_sentence_complexity,
						compositional=compositional, compositional_eval=compositional_eval,
						compositional_holdout=compositional_holdout,
						test_sentence=test_sentence,
						)
	sim.reset()
	# insert info into cfg
	import envs.language.cfg as langcfg
	langcfg.configurations['num_actions'] = sim.num_actions
	langcfg.configurations['action_dict'] = sim.action_dict
	env = langenv.EnvWrapper(sim)
	# add acme wrappers
	wrapper_list = [
		acme_wrappers.ObservationActionRewardWrapper, # put action + reward in observation
		acme_wrappers.SinglePrecisionWrapper, # cheaper to do computation in single precision
	]
	return acme_wrappers.wrap_all(env, wrapper_list), sim


def main(
		eval_lvls, 
		lvls, 
		print_end_assbdict,
		spacing,
		compositional, 
		compositional_eval, compositional_holdout,
		groupname, searchname='initial', 
		nrepeats=100,
		):
	'''
		lvls: list[int]
			list of puzzle_num_blocks to evaluate, should be within range [2, puzzle_max_blocks]
		compositional: bool
			if True, evaluating using compositional holdout
			if False, evaluating using specified lvl for puzzle_num_blocks
		compositional_eval: bool
	compositional_holdout: list
		groupname: str
		searchname: str
		nrepeats: int
			num of episode samples for random and expert demo
	'''
	default_log_dir = os.environ['RL_RESULTS_DIR']
	base_dir = os.path.join(default_log_dir, searchname, groupname,)
	seed_path, env_kwargs, agent_kwargs = load_settings(base_dir=base_dir, run='.')
	config = muzero.Config(**agent_kwargs)

	def tmp_obs_encoder(
		inputs: acme_wrappers.observation_action_reward.OAR,
		num_actions: int,
		max_sentence_length: int=langcfg.configurations['max_sentence_length'],
		num_fibers: int=langcfg.configurations['num_fibers'],
		num_areas: int=langcfg.configurations['num_areas'],
		max_assemblies: int=langcfg.configurations['max_assemblies'],
		num_pos: int=langcfg.configurations['num_pos'],
		num_words: int=langcfg.configurations['num_words'],):
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
			# print(f"gru_input shape {gru_input.shape} \noutput_sequence shape {output_sequence.shape} \ngoal_repr shape {goal_repr.shape}")
			# concatenate embeddings and previous reward and action
			x = jnp.concatenate((
				curlex_embed(jax.nn.one_hot(x.observation[:cut1], num_words).reshape(-1)),
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

	muzerotrainer.observation_encoder = lambda inputs, num_actions: tmp_obs_encoder(inputs=inputs,num_actions=num_actions)


	if eval_lvls and len(lvls)>0:
		modelsolved = [] # ratio of puzzles solved
		modelsolvedsem = [] 
		heuristicsolved = []
		heuristicsolvedsem = []
		randomsolved = []
		randomsolvedsem = []
		modelreward = [] # avg episode reward
		modelrewardsem = []
		heuristicreward = []
		heuristicrewardsem = []
		randomreward = []
		randomrewardsem = []
		modelsteps = [] # num of steps for solving a puzzle 
		modelstepssem = []
		heuristicsteps = [] # num of steps for solving a puzzle
		heuristicstepssem = []
		randomsteps = [] # num of steps for solving a puzzle
		randomstepssem = []
		for lvl in lvls:
			env, sim = make_test_environment(
											evaluation=True,
											spacing=spacing,
											eval_sentence_complexity=lvl,
											compositional=compositional, compositional_eval=compositional_eval,
											compositional_holdout=compositional_holdout,
											test_sentence=None,
											)
			mcts_policy = functools.partial(mctx.gumbel_muzero_policy,max_depth=config.max_sim_depth,num_simulations=config.num_simulations,gumbel_scale=config.gumbel_scale)
			discretizer = utils.Discretizer(num_bins=config.num_bins,max_value=config.max_scalar_value,tx_pair=config.tx_pair,)
			builder = basics.Builder(
				config=config,
				get_actor_core_fn=functools.partial(muzero.get_actor_core,evaluation=True,mcts_policy=mcts_policy,discretizer=discretizer,),
				ActorCls=functools.partial(basics.BasicActor, observers=[muzerotrainer.MuObserver(period=100000)],),
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
			load_outputs = load_agent(env=env, config=config, builder=builder, network_factory=network_factory, seed_path=seed_path, use_latest=True, evaluation=True)
			reload(load_outputs.checkpointer, seed_path) # can use this to load in latest checkpoints
			action_dict = langcfg.configurations['action_dict']

			print(f"\n----------------------- Evaluating lvl {lvl}")
			print(f"Muzero {groupname}")
			lvlsteps = [] # num steps for successful puzzles
			lvlepsr = [] # episode reward
			lvlsolved = [] # whether the puzzle is solved
			for irepeat in range(nrepeats):
				print(f"irepeat {irepeat}")
				num_words, goal_lex, goal_pos = langutils.sample_episode(
													difficulty_mode=lvl, 
													cur_curriculum_level=None, 
													max_complexity=langcfg.configurations['max_complexity'], 
													max_sentence_length=langcfg.configurations['max_sentence_length'],
													spacing=spacing, 
													compositional=compositional, 
													compositional_eval=compositional_eval, 
													compositional_holdout=compositional_holdout,
															)
				env, sim = make_test_environment(
											evaluation=True,
											spacing=None,
											eval_sentence_complexity=None,
											compositional=None, compositional_eval=None,
											compositional_holdout=None,
											test_sentence=[goal_lex, goal_pos],
											)
				load_outputs = load_agent(env=env, config=config, builder=builder, network_factory=network_factory, seed_path=seed_path, use_latest=True, evaluation=True)
				reload(load_outputs.checkpointer, seed_path)
				actor = load_outputs.actor
				timestep = env.reset()
				actor.observe_first(timestep)
				ends = False
				t = 0
				epsr = 0
				actions = []
				while not ends:
					action = actor.select_action(timestep.observation)
					actions.append(int(action))
					timestep = env.step(action)
					steptype = timestep.step_type
					r = timestep.reward
					state = timestep.observation[0]
					t += 1 # increment time step
					epsr += r # episode cumulative reward
					if steptype==2: # if final timestep
						ends = True
						if (t<langcfg.configurations['max_steps']-1) or (epsr>0.85):
							lvlsolved.append(1) # terminates early or solved at max step
							lvlsteps.append(t)
						else:
							lvlsolved.append(0)
					print(f"\tt={t}, action_name: {action_dict[int(action)]}, r={round(float(r),5)}, ends={ends}")
				if print_end_assbdict>0 and random.random()<print_end_assbdict:
					assbdict, lastassb = langutils.get_end_assembly_dict(sim, actions)
					print(f"Goal words {goal_lex}\nGoal pos {goal_pos}")
					pprint.pprint(assbdict)
					pprint.pprint(lastassb)
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

			print(f"Heuristic")
			lvlsolved = []
			lvlepsr = []
			lvlsteps = [] # num of expert steps
			for irepeat in range(nrepeats):
				print(f"irepeat {irepeat}")
				state, _ = sim.reset()
				epsr = 0
				expert_demo = langutils.expert_demo_language(sim)
				for t, a in enumerate(expert_demo):
					state, r, terminated, truncated, _ = sim.step(a)
					epsr += r # episode cumulative reward
					print(f"\tt={t}, action_name: {action_dict[int(a)]}, r={round(float(r),5)}")
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
		\nmodelsteps={modelsteps}\nmodelstepssem={modelstepssem}\
		\nheuristicsolved={heuristicsolved}\nheuristicsolvedsem={heuristicsolvedsem}\
		\nheuristicsteps={heuristicsteps}\nheuristicstepssem={heuristicstepssem}\
		\nrandomsolved={randomsolved}\nrandomsolvedsem={randomsolvedsem}\
		\nrandomsteps={randomsteps}\nrandomstepssem={randomstepssem}")

'''
salloc -p gpu_test -t 0-03:00 --mem=80000 --gres=gpu:1

salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

python configs/language/eval_muzero.py 
'''

if __name__ == "__main__":
	random.seed(0)
	main(
		eval_lvls=True, lvls=[2], # whether to eval on varying complexity (num words)
		print_end_assbdict=0.1, # whether to sample and print assembly dict at the end of an episode
		nrepeats=50, # num samples for all analyses except eval_steps
		groupname='Mgru2+comp-5v.7nospace', # model to load
		spacing=False,
		compositional=True, # whether the training setting is compositional
		compositional_eval=False, # whether to eval on comp holdout
		compositional_holdout=[
[-1,'adj','noun', 'intransverb', -1,-1,-1, -1,-1,-1, 'adv'],
						
['det',-1,'noun', 'transverb', -1,'adj','noun', -1,-1,-1,-1], 
[-1,-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
[-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', -1],

['det',-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
[-1,'adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
[-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
[-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],

['det','adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
[-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
[-1,-1,'noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
['det','adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],

['det','adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun',-1],
[-1,'adj','noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
[-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun','adv'],
[-1,-1,'noun', 'transverb', 'det','adj','noun', 'prep','det','noun',-1],
[-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun','adv'],

],
			)

'''

aa
'''