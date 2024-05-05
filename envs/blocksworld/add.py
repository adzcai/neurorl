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
				max_steps = cfg['max_steps'],
				action_cost = cfg['action_cost'],
				reward_decay_factor = cfg['reward_decay_factor'],
				maxnprocesses = cfg['maxnprocesses'],
				verbose=False):
		super().__init__(max_steps = max_steps,
						action_cost = action_cost,
						reward_decay_factor = reward_decay_factor,
						puzzle_max_blocks = 10, # TODO
						verbose = verbose)
		self.maxnprocesses = maxnprocesses
		assert cfg['cfg'] == 'add', f"cfg is {cfg['cfg']}"

	def __create_parse_goal(self, shuffle=True, cur_curriculum_level=-1):
		# cur_curriculum_level: {-1, 1,2,...,max_stack_blocks}
		goal = [None] * self.stack_max_blocks # dummy goal template, to be filled
		num_blocks = None
		if cur_curriculum_level==-1:
			num_blocks = random.randint(1, self.stack_max_blocks)
		else:
			cur_curriculum_level = max(cur_curriculum_level, 1)
			population = list(range(1, self.stack_max_blocks+1))
			weights = np.zeros_like(population, dtype=np.float32)
			weights[cur_curriculum_level-1] += 0.7 # weight for current level
			weights[max(cur_curriculum_level-2, 0)] += 0.15 # weight for the prev level
			weights[: max(cur_curriculum_level-2, 1)] += 0.15 / max(cur_curriculum_level-2, 1) # weight for easier level
			num_blocks = random.choices(population=population, weights=weights, k=1)[0]
		assert num_blocks <= self.stack_max_blocks, \
			f"number of actual blocks to parse {num_blocks} should be smaller than stack_max_blocks {self.stack_max_blocks}"
		stack = random.sample(list(range(self.puzzle_max_blocks)), num_blocks) # the actual blocks in the stack
		if shuffle:
			random.shuffle(stack)
		goal[:num_blocks] = stack
		return num_blocks, goal

	def __curriculum(self, cur_curriculum_level=-1):
		# sample num processes given current level
		if cur_curriculum_level==-1: # uniform
			return random.randint(0, self.maxnprocesses)
		population = list(range(self.maxnprocesses+1))
		weights = np.zeros_like(population, dtype=np.float32)
		weights[cur_curriculum_level] += 0.7 # weight for current level
		weights[max(cur_curriculum_level-1, 0)] += 0.15 # weight for the prev level
		weights[: max(cur_curriculum_level-1, 1)] += 0.15 / max(cur_curriculum_level-1, 1) # weight for easier level
		assert np.sum(weights)==1, f"weights {weights} should sum to 1"
		nprocesses = random.choices(population=population, weights=weights, k=1)[0]
		return nprocesses

	def reset(self, shuffle=True, difficulty_mode='uniform', cur_curriculum_level=-1):
		'''
		Reset environment for new episode.
		Input:
			shuffle: (boolean=True)
				whether to shuffle parse goal (parse goal will be of uniform num of blocks)
			difficulty_mode: {'uniform', 'curriculum', int}
				first remove/add arbitrary num of blocks from parsed goal
				then the difficulty_mode determines number of processes (add/remove) before creating the final goal
			cur_curriculum_level: {None, 0, 1, ..., self.maxnprocesses}
				if not None, determines number of blocks added after remove
		Return:
			state: (numpy array with float32)
			info: (any=None)
		'''
		import envs.blocksworld.cfg as config
		cur_curriculum_level = config.configurations['add']['cur_curriculum_level']
		self.num_blocks, self.goal = self.__create_parse_goal(shuffle=shuffle, cur_curriculum_level=cur_curriculum_level)
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.num_fibers = len(self.stateidx_to_fibername.keys())
		self.just_projected = False # record if the previous action was project
		self.all_correct = False # if the most recent readout has everything correct
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are ever correct in the episode
		self.current_time = 0 # current step in the episode
		self.num_assemblies = self.puzzle_max_blocks
		# first parse the stack
		parse_actions = utils.expert_demo_parse(self.goal, self.num_blocks)
		for t, a in enumerate(parse_actions):
			self.state, r, terminated, truncated, info = super().step(a)
		# decide number of add/remove before returning final goal
		nprocesses = random.randint(0, self.maxnprocesses) if self.num_blocks!=self.stack_max_blocks else random.randint(1, self.maxnprocesses)
		if difficulty_mode == 'curriculum':
			assert type(cur_curriculum_level)==int, f"difficulty_mode is curriculum but cur_curriculum_level is {cur_curriculum_level}"
			nprocesses = self.__curriculum(cur_curriculum_level)
		elif difficulty_mode != 'uniform' and type(difficulty_mode)==int:
			assert difficulty_mode>=0, f"difficulty_mode {difficulty_mode} should be int>=0"
			nprocesses = difficulty_mode
		if (self.num_blocks==self.stack_max_blocks) and (nprocesses==0):
			nprocesses = 1 # full stack, need at least 1 process
			# print(f'self.num_blocks==self.stack_max_blocks and nprocesses=0, adjusted nprocesses to 1')
		# do n rounds of add/remove 
		for iproc in range(nprocesses): 
			curprocess = random.choice(['add', 'remove'])
			if self.num_blocks==0:	# check remaining number of blocks in stack
				curprocess = 'add' # empty stack, can only add
			elif self.num_blocks==self.stack_max_blocks: 
				curprocess = 'remove' # full stack, can only remove
			elif (self.num_blocks==self.stack_max_blocks-1) and (iproc==nprocesses-1):
				curprocess = 'remove' # last round should not produce full stack
			if curprocess == 'remove':
				self.goal = self.goal[1:] + [None] # goal stack after removal
				self.just_projected = False # reset
				self.all_correct = False # reset most recent readout correct record
				self.correct_record = np.zeros_like(self.goal) # reset episode record
				self.current_time = 0 # current step in the episode
				for ib in range(self.stack_max_blocks): # update state vector to encode new goal
					if self.goal[ib]==None:  # filler for empty block
						self.state[self.area_to_stateidx['goal_stack'][ib]] = -1
					else:
						self.state[self.area_to_stateidx['goal_stack'][ib]] = self.goal[ib]
				remove_actions = utils.expert_demo_remove(self) # actions to remove this block
				# print(f"\tremoving for goal {self.goal}, remove_actions {remove_actions}")
				for t, a in enumerate(remove_actions): # perform remove
					self.state, r, terminated, truncated, info = super().step(a)
				self.num_blocks -= 1 # update remaining num of blocks
				if self.num_blocks==0: # check whether need to clean state
					# print("\twipe!")
					self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
					self.num_assemblies = self.puzzle_max_blocks
			elif curprocess=='add':
				newblock = None
				while newblock==None: # sample a valid new block id
					tmpblock = random.randint(0, self.puzzle_max_blocks-1)
					if tmpblock not in self.goal: 
						newblock = tmpblock
				self.newblock = newblock
				self.goal = [newblock] + self.goal[:-1] # new goal after adding
				self.just_projected = False # reset
				self.all_correct = False # reset
				self.correct_record = np.zeros_like(self.goal) # reset
				self.current_time = 0 # reset
				for ib in range(self.stack_max_blocks): # update state vector to encode new goal
					if self.goal[ib]==None:  # filler for empty block
						self.state[self.area_to_stateidx['goal_stack'][ib]] = -1
					else:
						self.state[self.area_to_stateidx['goal_stack'][ib]] = self.goal[ib]
				add_actions = utils.expert_demo_add(self) # actions to add new block
				# print(f"adding for goal {self.goal}, \n\tadd_actions {add_actions}")
				for t, a in enumerate(add_actions):
					self.state, r, terminated, truncated, info = super().step(a)
					# print(f"\tt={t}, a={self.action_dict[a]}, r={r}, state={self.state}, terminated={terminated}")
				self.num_blocks += 1 # update remaining number of blocks stack
		# the real goal is adding 1 block to the current stack
		assert self.num_blocks+1 <= self.stack_max_blocks, f"num blocks after add {self.num_blocks+1} exceeds stack max blocks {self.stack_max_blocks}"
		newblock = None # sample a valid new block id 
		while newblock==None: 
			tmpblock = random.randint(0, self.puzzle_max_blocks-1)
			if tmpblock not in self.goal: 
				newblock = tmpblock
		self.newblock = newblock
		self.goal = [newblock] + self.goal[:-1] # new goal
		for ib in range(self.stack_max_blocks): # update state vector to encode new goal
			if self.goal[ib]==None:  # filler for empty block
				self.state[self.area_to_stateidx['goal_stack'][ib]] = -1
			else:
				self.state[self.area_to_stateidx['goal_stack'][ib]] = self.goal[ib]
		self.correct_record = np.zeros_like(self.goal) # reset 
		self.current_time = 0 # reset 
		self.just_projected = False # reset 
		self.all_correct = False # reset
		info = None
		# print(f"final add goal ready: {self.goal}")
		return self.state.copy(), info




def test_simulator(expert=True, repeat=10, verbose=False):
	sim = Simulator(verbose=verbose)
	pprint.pprint(sim.action_dict)
	avg_expert_len = []
	for difficulty in range(sim.maxnprocesses+1):
		expert_len = []
		for r in range(repeat):
			state, _ = sim.reset(shuffle=True, difficulty_mode="curriculum", cur_curriculum_level=difficulty)
			print(f'------------ repeat {r}, state after reset\t{state}')
			expert_demo = utils.expert_demo_add(sim) if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			print(f"expert_demo: {expert_demo}")  if verbose else 0
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				# print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated},\n\tjust_projected={sim.just_projected}, all_correct={sim.all_correct}, correct_record={sim.correct_record}')  if verbose else 0
				# print(f'\tnext state {next_state}\t')  if verbose else 0
			readout = utils.synthetic_readout(sim.assembly_dict, sim.last_active_assembly, sim.head, len(sim.goal), sim.blocks_area)
			print(f'end of episode (difficulty={difficulty}), num_blocks={sim.num_blocks}, synthetic readout {readout}, goal {sim.goal}, total reward={rtotal}')  if verbose else 0
			if expert:
				assert readout == sim.goal, f"readout {readout} and goal {sim.goal} should be the same"
				assert terminated, "episode should be done"
				assert np.isclose(rtotal, sim.episode_max_reward-sim.action_cost*nsteps), \
						f"rtotal {rtotal} and theoretical total {sim.episode_max_reward-sim.action_cost*nsteps} should be roughly the same"
				expert_len.append(len(expert_demo))
		avg_expert_len.append(np.mean(expert_len)) if expert else 0
	print(f"\n\navg expert demo length {avg_expert_len}\n\n")




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
		min_val, max_val = space.min(), configurations['parse']['max_assemblies']
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
		sim = Simulator()
		return EnvWrapper(sim)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()

if __name__ == "__main__":
	random.seed(0)
	test_simulator(expert=False, repeat=2000, verbose=False)
	test_simulator(expert=True, repeat=1000, verbose=False)
	
	absltest.main()

