import numpy as np
import random
import pprint

from envs.blocksworld import utils
from envs.blocksworld.cfg import configurations
import envs.blocksworld.parse as parse

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree

cfg = configurations['add']

class Simulator(parse.Simulator):
	def __init__(self, 
				max_blocks = cfg['max_blocks'],
				max_steps = cfg['max_steps'],
				action_cost = cfg['action_cost'],
				reward_decay_factor = cfg['reward_decay_factor'],
				episode_max_reward = cfg['episode_max_reward'],
				verbose=False):
		super().__init__(max_blocks = max_blocks,
						max_steps = max_steps,
						action_cost = action_cost,
						reward_decay_factor = reward_decay_factor,
						episode_max_reward = episode_max_reward,
						verbose = verbose)
		assert cfg['cfg'] == 'add', f"cfg is {cfg['cfg']}"



	def __create_parse_goal(self, shuffle=True):
		# assuming uniform parse goal 
		goal = [None] * self.max_blocks # dummy goal template, to be filled
		num_blocks = random.randint(1, self.max_blocks)
		assert num_blocks <= self.max_blocks, \
			f"number of actual blocks to parse {num_blocks} should be smaller than max_blocks {self.max_blocks}"
		stack = list(range(num_blocks)) # the actual blocks in the stack, to be filled
		if shuffle:
			random.shuffle(stack)
		goal[:num_blocks] = stack
		return num_blocks, goal

	def __add_curriculum(self, curriculum):
		return min(max(self.max_blocks-self.num_blocks-1, 0), curriculum)

	def reset(self, shuffle=True, difficulty_mode='uniform', cur_curriculum_level=None):
		'''
		Reset environment for new episode.
		Input:
			shuffle: (boolean=True)
				whether to shuffle parse goal (parse goal will be of uniform length)
			difficulty_mode: {'uniform', 'curriculum'}
				first remove uniform number of blocks from parsed goal
				then the difficulty_mode determines number of blocks added
			cur_curriculum_level: {None, 0, 1, ..., self.max_blocks-1}
				if not None, determines number of blocks added after remove
		Return:
			state: (numpy array with float32)
			info: (any=None)
		'''
		self.num_blocks, self.goal = self.__create_parse_goal(shuffle=shuffle)
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.num_fibers = len(self.stateidx_to_fibername.keys())
		self.just_projected = False # record if the previous action was project
		self.all_correct = False # if the most recent readout has everything correct
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are ever correct in the episode
		self.current_time = 0 # current step in the episode
		self.num_assemblies = self.max_blocks
		# first parse the stack
		parse_actions = utils.parse_expert_demo(self.goal, self.num_blocks)
		print(f"\n\nparsing {self.goal}...") 
		for t, a in enumerate(parse_actions):
			self.state, r, terminated, truncated, info = super().step(a)
		# then remove arbitrary number of blocks from the stack
		nremove = random.randint(0, self.num_blocks) if self.num_blocks<self.max_blocks else random.randint(1, self.num_blocks)
		for ith in range(nremove): # remove blocks one by one
			self.goal = self.goal[1:] + [None] # goal stack after removal
			self.just_projected = False # record if the previous action was project
			self.all_correct = False # if the most recent readout has everything correct
			self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are ever correct in the episode
			self.current_time = 0 # current step in the episode
			for ib in range(self.max_blocks): # update state vector to encode new goal
				if self.goal[ib]==None:  # filler for empty block
					self.state[self.area_to_stateidx['goal_stack'][ib]] = -1
				else:
					self.state[self.area_to_stateidx['goal_stack'][ib]] = self.goal[ib]
			remove_actions = utils.remove_expert_demo(self) # actions to remove this block
			print(f"\tremoving for goal {self.goal}, remove_actions {remove_actions}")
			for t, a in enumerate(remove_actions):
				self.state, r, terminated, truncated, info = super().step(a)
		self.num_blocks -= nremove # update number of blocks in current stack
		if self.num_blocks==0: # TODO: need to check why this is needed
			print("\twipe!")
			self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
			self.num_assemblies = self.max_blocks
		# then add arbitrary number of blocks to the stack
		nadd = random.randint(0, max(self.max_blocks-self.num_blocks-1, 0)) if difficulty_mode=='uniform' else self.__add_curriculum(cur_curriculum_level)
		for ith in range(nadd): # add blocks one by one
			newblock = None
			while newblock==None: # sample a valid new block id
				tmpblock = random.randint(0, self.max_blocks-1)
				if tmpblock not in self.goal: 
					newblock = tmpblock
			self.newblock = newblock
			self.goal = [newblock] + self.goal[:-1] # new goal after adding
			self.just_projected = False # record if the previous action was project
			self.all_correct = False # if the most recent readout has everything correct
			self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are ever correct in the episode
			self.current_time = 0 # current step in the episode
			for ib in range(self.max_blocks): # update state vector to encode new goal
				if self.goal[ib]==None:  # filler for empty block
					self.state[self.area_to_stateidx['goal_stack'][ib]] = -1
				else:
					self.state[self.area_to_stateidx['goal_stack'][ib]] = self.goal[ib]
			add_actions = utils.add_expert_demo(self) # actions to remove this block
			print(f"adding for goal {self.goal}, \n\tadd_actions {add_actions}")
			for t, a in enumerate(add_actions):
				self.state, r, terminated, truncated, info = super().step(a)
				print(f"\tt={t}, a={self.action_dict[a]}, r={r}, state={self.state}, terminated={terminated}")
		self.num_blocks += nadd # update number of blocks in current stack
		# the real goal is adding 1 block to the current stack
		assert self.num_blocks+1 <= self.max_blocks, f"num blocks after add {self.num_blocks+1} exceeds max blocks {self.max_blocks}"
		newblock = None # sample a valid new block id 
		while newblock==None: 
			tmpblock = random.randint(0, self.max_blocks-1)
			if tmpblock not in self.goal: 
				newblock = tmpblock
		self.newblock = newblock
		self.goal = [newblock] + self.goal[:-1] # new goal
		for ib in range(self.max_blocks): # update state vector to encode new goal
			if self.goal[ib]==None:  # filler for empty block
				self.state[self.area_to_stateidx['goal_stack'][ib]] = -1
			else:
				self.state[self.area_to_stateidx['goal_stack'][ib]] = self.goal[ib]
		self.correct_record = np.zeros_like(self.goal) # reset episode correct record
		self.current_time = 0 # reset episode timer
		self.just_projected = False # reset 
		self.all_correct = False # reset
		info = None
		print(f"final add goal ready: {self.goal}")
		return self.state.copy(), info




def test_simulator(max_blocks=7, expert=True, repeat=10, verbose=False):
	sim = Simulator(max_blocks=max_blocks, verbose=verbose)
	pprint.pprint(sim.action_dict)
	for difficulty in range(max_blocks+1):
		for _ in range(repeat):
			print(f'------------ repeat {repeat}, state after reset\t{sim.reset(shuffle=True, difficulty_mode="curriculum", cur_curriculum_level=difficulty)[0]}')
			expert_demo = utils.add_expert_demo(sim) if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			print(f"expert_demo: {expert_demo}")
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated},\n\tjust_projected={sim.just_projected}, all_correct={sim.all_correct}, correct_record={sim.correct_record}')
				print(f'\tnext state {next_state}\t')
			readout = utils.synthetic_readout(sim.assembly_dict, sim.last_active_assembly, sim.head, len(sim.goal), sim.blocks_area)
			print(f'end of episode (difficulty={difficulty}), num_blocks={sim.num_blocks}, synthetic readout {readout}, goal {sim.goal}, total reward={rtotal}')
			if expert:
				assert readout == sim.goal, f"readout {readout} and goal {sim.goal} should be the same"
				assert terminated, "episode should be done"
				assert np.isclose(rtotal, sim.episode_max_reward-sim.action_cost*nsteps), \
						f"rtotal {rtotal} and theoretical total {sim.episode_max_reward-sim.action_cost*nsteps} should be roughly the same"




class EnvWrapper(dm_env.Environment):
	'''
	Wraps a Simulator object to be compatible with dm_env.Environment
	Reference: 
		https://github.com/wcarvalho/human-sf/blob/da0c65d04be708199ffe48d5f5118b295bfd43a3/lib/dm_env_wrappers.py#L15
		https://github.com/google-deepmind/dm_env/
		https://github.com/google-deepmind/acme/
	'''
	def __init__(self, environment: Simulator, shuffle=True, difficulty_mode='uniform', cur_curriculum_level=None):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		self._environment.reset()
		obs_space = self._environment.state
		act_space = self._environment.num_actions-1 # maximum action index
		self._observation_spec = _convert_to_spec(obs_space, name='observation')
		self._action_spec = _convert_to_spec(act_space, name='action')
		self.shuffle = shuffle # whether to shuffle blocks for each episode
		self.difficulty_mode = difficulty_mode
		self.cur_curriculum_level = cur_curriculum_level

	def reset(self) -> dm_env.TimeStep:
		self._reset_next_step = False
		observation, info = self._environment.reset(shuffle=self.shuffle, difficulty_mode=self.difficulty_mode, cur_curriculum_level=self.cur_curriculum_level)
		self._last_info = info
		return dm_env.restart(observation)
	
	def step(self, action: types.NestedArray) -> dm_env.TimeStep:
		if self._reset_next_step:
			return self.reset()
		observation, reward, done, truncated, info = self._environment.step(action)
		self._reset_next_step = done or truncated
		self._last_info = info
		# Convert the type of the reward based on the spec, respecting the scalar or array property.
		reward = tree.map_structure(
			lambda x, t: (  # pylint: disable=g-long-lambda
				t.dtype.type(x)
				if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
			reward,
			self.reward_spec())
		if truncated:
			return dm_env.truncation(reward, observation)
		if done:
			return dm_env.termination(reward, observation)
		return dm_env.transition(reward, observation)
	
	def observation_spec(self) -> types.NestedSpec:
		return self._observation_spec

	def action_spec(self) -> types.NestedSpec:
		return self._action_spec

	def get_info(self) -> Optional[Dict[str, Any]]:
		return self._last_info

	@property
	def environment(self) -> Simulator:
		return self._environment

	def __getattr__(self, name: str):
		if name.startswith('__'):
			raise AttributeError('attempted to get missing private attribute {}'.format(name))
		return getattr(self._environment, name)

	def close(self):
		self._environment.close()
		




def _convert_to_spec(space: Any,
					name: Optional[str] = None) -> types.NestedSpec:
	"""
	Converts a Python data structure to a dm_env spec or nested structure of specs.
	The function supports scalars, numpy arrays, tuples, and dictionaries.
	Args:
		space: The data item to convert (can be scalar, numpy array, tuple, or dict).
		name: Optional name to apply to the return spec.
	Returns:
		A dm_env spec or nested structure of specs, corresponding to the input item.
	"""
	if isinstance(space, int): # scalar int for max idx of an action
		dtype = type(space)
		min_val = 0 # minimum action index (inclusive)
		max_val = space # maximum action index (inclusive)
		try:
			assert name=='action'
		except:
			raise ValueError('Converting integer to dm_env spec, but name is not action')
		return specs.DiscreteArray(
			num_values=max_val+1,
			name=name
		)
	elif isinstance(space, np.ndarray): # observation
		min_val, max_val = space.min(), cfg['max_assemblies']
		try:
			assert name=='observation'
		except:	
			raise ValueError("Converting np.ndarray to dm_env spec, but name is not 'observation'")
		return specs.BoundedArray(
			shape=space.shape,
			dtype=space.dtype,
			minimum=min_val,
			maximum=max_val,
			name=name
		)
	elif isinstance(space, tuple):
		return tuple(_convert_to_spec(s, name) for s in space)
	elif isinstance(space, dict):
		return {
			key: _convert_to_spec(value, key)
			for key, value in space.items()
		}
	else:
		raise ValueError('Unsupported data type for conversion to dm_env spec: {}'.format(space))
	

class Test(test_utils.EnvironmentTestMixin, absltest.TestCase):
	def make_object_under_test(self):
		sim = Simulator(max_blocks=7)
		return EnvWrapper(sim)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()

if __name__ == "__main__":
	# random.seed(1)
	test_simulator(max_blocks=7, expert=True, repeat=100, verbose=False)
	
	absltest.main()

