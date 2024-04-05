'''
learn to plan in blocksworld using neural actions parse, add, remove
suppose we have 3 segments in the state: current stacks, table stacks, goal stacks
	(along with other info, e.g. pointer indices, correct history, etc.)
'''
import random
import numpy as np
from envs.blocksworld.cfg import configurations
from envs.blocksworld import utils

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree

cfg = configurations['plan']

'''
TODO
cur stack and goal stack are: top (must have) -> bottom (optional)
expert demo input and goal
state vec 1D
base_block_reward -> unit reward
'''

class Simulator():
	def __init__(self,
			  puzzle_max_stacks=cfg['puzzle_max_stacks'], 
			  puzzle_max_blocks=cfg['puzzle_max_blocks'], 
			  stack_max_blocks=configurations['stack_max_blocks'], 
			  max_steps=cfg['max_steps'],
			  episode_max_reward=configurations['episode_max_reward'],
			  reward_decay_factor=cfg['reward_decay_factor'],
			  action_cost=cfg['action_cost'],
			  verbose=False):
		assert cfg['cfg'] == 'plan', f"cfg is {cfg['cfg']}"
		self.puzzle_max_stacks = puzzle_max_stacks
		self.puzzle_max_blocks = puzzle_max_blocks
		self.stack_max_blocks = stack_max_blocks
		self.episode_max_reward = episode_max_reward
		self.reward_decay_factor = reward_decay_factor 
		self.action_cost = action_cost
		self.max_steps = max_steps # maximum number of actions allowed in an episode
		self.verbose = verbose
		self.action_dict = self.create_action_dictionary() 
		self.num_actions = len(self.action_dict)


	def close(self):
		del state
		del action_dict, action_to_statechange
		del current_time
		return 
	

	def reset(self, difficulty=None, curriculum=cfg['curriculum']):
		info = None
		self.puzzle_num_blocks, input_stacks, goal_stacks = utils.sample_random_puzzle(puzzle_max_stacks=self.puzzle_max_stacks, 
													puzzle_max_blocks=self.puzzle_max_blocks, stack_max_blocks=self.stack_max_blocks
													difficulty=difficulty, curriculum=curriculum)
		assert difficulty==None or (difficulty == self.puzzle_num_blocks)
		# format the input and goal to same length
		self.goal_stacks = [[-1 for _ in range(self.stack_max_blocks)] for _ in range(puzzle_max_stacks)] # [puzzle_max_stacks, stack_max_blocks]
		for istack in range(len(goal_stacks)): # a stack [-1, -1, ..., top block, ..., bottom block]
			for jblock in range(1, len(goal_stacks[istack])+1): # traverse backwards
				self.goal_stacks[istack][-jblock] = goal_stacks[istack][-jblock] 
		self.flipped_goal_stacks = self.__flip(self.goal_stacks)
		self.input_stacks = [[-1 for _ in range(self.stack_max_blocks)] for _ in range(puzzle_max_stacks)] # [puzzle_max_stacks, stack_max_blocks]
		for istack in range(len(input_stacks)): # a stack [-1, -1, ..., top block, ..., bottom block]
			for jblock in range(1, len(input_stacks[istack])+1): # traverse backwards
				self.input_stacks[istack][-jblock] = input_stacks[istack][-jblock]

		self.current_time = 0 # reset time 
		self.state, self.action_to_statechange = self.create_state_representation() # reset state
		return self.state.copy(), info
	
	def __update_state(self, changes):
		# set state idxs to new vals
		idxs, vals = changes[0], changes[1]
		for idx, val in zip(idxs, vals):
			self.state[idx] = val

	def __add_to_state(self, changes):
		# add vals to state idxs
		idxs, vals = changes[0], changes[1]
		for idx, val in zip(idxs, vals):
			self.state[idx] += val

	def __stack_empty(self, istart, length=self.stack_max_blocks):
		# all empty (-1) blocks in a stack in state
		return np.all(self.state[istart: istart+length] == -1)

	def __stack_full(self, istart, length=self.stack_max_blocks):
		# no empty (-1) blocks in a stack in state
		return np.all(self.state[istart: istart+length] != -1)
	
	def __pop_top_block(self, istart, length=self.stack_max_blocks):
		# remove the top block from a stack in the state 
		b = self.state[istart] # the top block to be popped
		remaining = self.state[istart+1 : istart+length] # the remaining blocks
		self.state[istart: istart+length-1] = remaining # shift everything to the left
		self.state[istart+length-1] = -1 # fill the last position with empty
		return b

	def __insert_top_block(self, block, istart, length=self.stack_max_blocks):
		# add the top block to a stack in the state 
		remaining = self.state[istart: istart+length-1] # the remaining blocks
		self.state[istart+1: istart+length] = remaining # shift everything to the right
		self.state[istart] = block # fill the first position with new block
	
	def __ith_empty_table(self, istart, length=self.puzzle_max_blocks, i=1):
		# find the state idx corresponding to the ith empty table 
		# assuming table size 1
		for idx in range(istart, istart+length):
			if self.state[idx] == -1:
				i -= 1
			if i == 0:
				return idx




	def step(self, action_idx):
		action_idx = int(action_idx)
		action_name = self.action_dict[action_idx] 
		action_to_statechange = self.action_to_statechange
		cur_pointer = self.state[action_to_statechange['stack_pointer']]
		table_pointer = self.state[action_to_statechange['table_pointer']]
		input_parsed = self.state[action_to_statechange['input_parsed']]
		goal_parsed = self.state[action_to_statechange['goal_parsed']]
		reward = -self.action_cost # default cost for performing any action
		terminated = False # whether the episode ended
		truncated = False # end due to max steps
		info = None
		if (not input_parsed) or (not goal_parsed):
			if (not input_parsed) and (not goal_parsed) and (action_name != "parse_input") and (action_name != "parse_goal"):
				reward -= self.action_cost*2  # BAD, perform any other actions before input or goal is parsed
				print('\tboth input and goal not parsed yet') if self.verbose else 0
			elif (not input_parsed) and action_name == "parse_input": # GOOD, parse input for first time
				self.__update_state(action_to_statechange[action_idx])
			elif (not goal_parsed) and action_name == "parse_goal": # GOOD, parse goal for first time
				self.__update_state(action_to_statechange[action_idx])
			else: # BAD, perform other actions when one of input or goal still not parsed
				reward -= self.action_cost*2
				print('\teither input or goal not parsed yet') if self.verbose else 0
		elif action_name == "next_stack":
			if cur_pointer + 1 >= self.puzzle_max_stacks: # BAD, new cur pointer idx out of range
				reward -= self.action_cost
				print('\tcur pointer out of range') if self.verbose else 0
			else: # GOOD, next stack
				self.__add_to_state(action_to_statechange[action_idx])
		elif action_name == "previous_stack":
			if cur_pointer == 0: # BAD, cur pointer is already minimum
				reward -= self.action_cost
				print('\tcur pointer out of range') if self.verbose else 0
			else: # GOOD, previous stack
				self.__add_to_state(action_to_statechange[action_idx])
		elif action_name == "next_table":
			if table_pointer + 1 >= self.max_blocks: # BAD, new table pointer out of range
				reward -= self.action_cost
				print('\ttable pointer out of range') if self.verbose else 0
			else: # GOOD, next table stack
				self.__add_to_state(action_to_statechange[action_idx])
		elif action_name == "previous_table":
			if table_pointer == 0: # BAD, table pointer is already minimum
				reward -= self.action_cost
				print('\ttable pointer out of range') if self.verbose else 0
			else: # GOOD, previous table stack
				self.__add_to_state(action_to_statechange[action_idx])
		elif action_name == "remove":
			if self.__stack_empty(action_to_statechange['cur_stacks_begin'][stack_pointer]): # BAD, cur stack is empty
				reward -= self.action_cost
				print('\tnothing to remove, cur stack empty') if self.verbose else 0
			else: # GOOD, pop the top block from cur stack
				block = self.__pop_top_block(action_to_statechange['cur_stacks_begin'][stack_pointer])
				sidx = self.__ith_empty_table(action_to_statechange['table'][0])
				self.__update_state([[sidx], [block]])
				print('\tremove top block', block_id) if self.verbose else 0
				r, terminated = self.__reward(curstacks=self.__flip(self.__decode_curstacks), goalstacks=self.flipped_goal_stacks)
				reward += r
		elif action_name == "add":
			if self.__stack_empty(action_to_statechange['table'][table_pointer], length=1): 
				reward -= self.action_cost # BAD, nothing to add, table stack is empty
				print('\tnothing to add, table stack empty') if self.verbose else 0
			elif self.__stack_full(action_to_statechange['cur_stacks_begin'][stack_pointer]): 
				reward -= self.action_cost # BAD, intent to add to cur stack, but stack full
				print('\tintend to add to full stack, last block in stack is',self.state[cur_pointer+self.max_blocks-1]) if self.verbose else 0
			else: # GOOD, add the block to cur stack
				new_block = self.__pop_top_block(action_to_statechange['table'][table_pointer], length=1) # pop from table
				self.__insert_top_block(new_block, action_to_statechange['cur_stacks_begin'][stack_pointer])
				r, terminated = self.__reward(curstacks=self.__flip(self.__decode_curstacks), goalstacks=self.flipped_goal_stacks)
				reward += r 
		elif action_name == "parse_input": # BAD, parse input repetitively
			assert input_parsed
			self.__update_state(action_to_statechange[action_idx])
			reward -= self.action_cost*2
			print('\tinput parsed again, reset') if self.verbose else 0
		elif action_name == "parse_goal": # BAD, parse goal repetitively
			self.__update_state(action_to_statechange[action_idx])
			reward -= self.action_cost*2
			print('\tgoal parsed again') if self.verbose else 0
		self.current_time += 1
		if self.current_time >= self.max_steps:
			truncated = True
		return self.state.copy(), reward, terminated, truncated, info


	def __intersection(self, curstack, goalstack):
		'''
		return the number of blocks that match in curstack and goalstack (starting from the bottom)
			and the number of blocks that need to be removed from curstack (i.e. the non matching blocks)
		'''
		intersection, num_to_remove = 0, 0
		for i in range(self.max_blocks-1, -1, -1): # iterate from bottom to top
			if goalstack[i] != -1 and (curstack[i]==goalstack[i]):
				intersection += 1
			else: # first nonmatching block
				break
		for j in range(self.max_blocks-1, -1, -1):
			if curstack[j]==-1: # find the idx of first empty block
				break
		num_to_remove = i-j # number of blocks to be removed from curstack
		return intersection, num_to_remove 
	
	def __flip(self, stacks):
		newstacks = []
		for s in stacks:
			news = []
			for ib in range(len(s)-1, -1, -1):
				if s[ib]==-1:
					continue
				else:
					news.append(s[ib])
			news += [-1] * (len(s)-len(news))
			newstacks.append(news)
		return newstacks

	def __decode_curstacks(self):
		curstacks = []
		for istack in range(self.puzzle_max_stacks):
			stack = []
			for jblock in range(self.stack_max_blocks):
				stack.append(int(self.state[self.action_to_statechange['cur_stacks_begin'][istack] + jblock]))
			curstacks.append(stack)
		return curstacks
		
	def __reward(self, curstacks, goalstacks):
		return 0

	def __readout_reward(self):
		cur_stacks = self.__decode_curstack()
		print('cur_stacks', cur_stacks) if self.verbose else 0
		intersection_record = self.state[self.max_blocks*(self.puzzle_max_stacks*2+1):self.max_blocks*(self.puzzle_max_stacks*2+1)+self.puzzle_max_stacks, :].tolist()
		base_block_reward = self.base_block_reward
		reward_decay_factor = self.reward_decay_factor
		score = 0
		intersections = [0] * self.puzzle_max_stacks # temporary intersection for the current readout
		for istack in range(self.puzzle_max_stacks): # iterate each stack
			intersections[istack], _ = self.__intersection(cur_stacks[istack], self.goal[istack]) # number of blocks that match 
			print('intersections', intersections, 'intersection_record', intersection_record) if self.verbose else 0
			if intersections[istack]!=0 and (intersection_record[istack][intersections[istack]-1]==0): # new match
				score += sum([base_block_reward / (reward_decay_factor**(d+1)) for d in range(intersections[istack])])
				intersection_record[istack][intersections[istack]-1] = 1 # update the intersection record for the new match
				if intersections[istack] > 1: # fill preceeding intersection record with all 1s
					intersection_record[istack][:intersections[istack]] = [1 for _ in range(intersections[istack])]
			elif intersections[istack] != 0: # already matched, give smaller reward
				score += 0
		all_correct = sum(intersections) == self.num_valid_blocks # all blocks are matched
		self.state[self.max_blocks*(self.puzzle_max_stacks*2+1):self.max_blocks*(self.puzzle_max_stacks*2+1)+self.puzzle_max_stacks,:] = np.array(intersection_record)
		return score, all_correct
	
		
	def create_state_representation(self):
		'''
		Initialize environment state for the new episode.
		Assuming self.input_stacks and self.goal_stacks are created already and matched in length.
		Assuming self.action_dict is created.
		A stack reads from left to right of the list as top block (must have) to bottom block (optional),
		which aligns with the ordering in parse/remove/add.
		E.g. top to bottom blocks in a stack |0|1|2|table --> stack [0,1,2,-1,-1,...,-1] of length stack_max_blocks
		Each table stack can only have max 1 block.
		There are puzzle_max_blocks table stacks available.
		Return:
			state: (1D list of int)
				[table stack 1 block, table stack 2 block, ..., table stack puzzle_max_blocks block,
				cur stack 1 top block, cur stack 1 second block, ..., cur stack 1 bottom block, cur stack 2 top block, ..., cur puzzle_max_stacks bottom block,
				goal stack 1 top block, goal stack 1 second block, ..., goal stack 1 bottom block, goal stack 2 top block, ..., goal puzzle_max_stacks bottom block,
				intersection record (num of correct blocks throughout episode) for stack 1, ..., intersection record for stack puzzle_max_stacks,
				cur stack pointer (int 0 to puzzle_max_stacks-1), table pointer (int 0 to puzzle_max_blocks-1),
				input parsed throughout episode (0 or 1), goal parsed throughout episode (0 or 1)
			action_to_statechange: (dict)
				{action_idx: [[state idxs to change], [new values in state]], ...
				info: [state idxs related to the info], ...}
				action_idx should match self.action_dict
		'''
		state_vec = []
		state_idx = 0
		action_idx = 0
		action_to_statechange = {}
		# action -> state change: parse input
		parse_input_action_idx = action_idx
		action_to_statechange[action_idx] = [[],[]] # if parse input called, where and what to update in state
		assert self.action_dict[action_idx] == 'parse_input'
		# info -> state idx: each cur stack top block to state location
		action_to_statechange['cur_stacks_begin'] = []
		# encode current stacks
		for istack in range(self.puzzle_max_stacks):
			action_to_statechange['cur_stacks_begin'].append(state_idx)
			for jblock in range(self.stack_max_blocks):
				state_vec.append(-1) # initialized as empty
				action_to_statechange[action_idx][0].append(state_idx)
				action_to_statechange[action_idx][1].append(self.input_stacks[istack][jblock])
				state_idx += 1
		action_idx += 1
		# action -> state change: parse goal
		parse_goal_action_idx = action_idx
		action_to_statechange[action_idx] = [[], []] # if parse goal called, where and what to update in state
		assert self.action_dict[action_idx] == 'parse_goal'
		# encode goal stacks
		for istack in range(self.puzzle_max_stacks):
			for jblock in range(self.stack_max_blocks):
				state_vec.append(-1) # initialized as empty
				action_to_statechange[action_idx][0].append(state_idx)
				action_to_statechange[action_idx][1].append(self.goal_stacks[istack][jblock])
				state_idx += 1
		action_idx += 1
		# info -> state idx: table block to state location
		action_to_statechange['table'] = [] # will be filled with state idxs for each table stack
		# encode table stacks
		for _ in range(self.puzzle_max_blocks):
			state_vec.append(-1) # initialized as empty
			action_to_statechange['table'].append(state_idx) 
			state_idx += 1
		# info -> state idx: intersection record of each stack to state location
		action_to_statechange['intersection'] = []
		# encode intersection record
		for _ in range(self.puzzle_max_stacks): 
			state_vec.append(0) # num correct blocks in each stack
			action_to_statechange['intersection'].append(state_idx)
			state_idx += 1
		# action -> state change: next cur stack
		action_to_statechange[action_idx] = [[], []]
		assert self.action_dict[action_idx] == 'next_stack'
		# encode cur stack pointer
		state_vec.append(0) # initialize to first stack
		action_to_statechange[action_idx][0].append(state_idx)
		action_to_statechange[action_idx][1].append(+1)
		action_idx += 1
		action_to_statechange[parse_input_action_idx][0].append(state_idx) # reset cur pointer when parse input
		action_to_statechange[parse_input_action_idx][1].append(0)
		# action -> state change: prev cur stack
		action_to_statechange[action_idx] = [[], []]
		assert self.action_dict[action_idx] == 'previous_stack'
		action_to_statechange[action_idx][0].append(state_idx)
		action_to_statechange[action_idx][1].append(-1)
		action_idx += 1
		# info -> state idx: flag for cur stack pointer to state location
		action_to_statechange['stack_pointer'] = state_idx
		state_idx += 1
		# action -> state change: next table stack
		action_to_statechange[action_idx] = [[], []]
		assert self.action_dict[action_idx] == 'next_table'
		# encode table pointer
		state_vec.append(0) # initialize to first stack
		action_to_statechange[action_idx][0].append(state_idx)
		action_to_statechange[action_idx][1].append(+1)
		action_idx += 1
		action_to_statechange[parse_input_action_idx][0].append(state_idx) # reset table pointer when parse input
		action_to_statechange[parse_input_action_idx][1].append(0)
		# action -> state change: prev table stack
		action_to_statechange[action_idx] = [[], []]
		assert self.action_dict[action_idx] == 'previous_table'
		action_to_statechange[action_idx][0].append(state_idx)
		action_to_statechange[action_idx][1].append(-1)
		action_idx += 1
		# info -> state idx: flag for table pointer to state location
		action_to_statechange['table_pointer'] = state_idx
		state_idx += 1
		# encode whether input parsed
		state_vec.append(0) # initialize to False
		# info -> state idx: flag for input parsed to state location
		action_to_statechange['input_parsed'] = state_idx
		action_to_statechange[parse_input_action_idx][0].append(state_idx) # update flag when parse input
		action_to_statechange[parse_input_action_idx][1].append(1)
		state_idx += 1
		# encode whether goal parsed
		state_vec.append(0) # initialize to False
		# info -> state idx: flag for goal parsed to state location
		action_to_statechange['goal_parsed'] = state_idx
		action_to_statechange[parse_goal_action_idx][0].append(state_idx) # update flag when parse goal
		action_to_statechange[parse_goal_action_idx][1].append(1)
		state_idx += 1
		return np.array(state_vec, dtype=np.float32), action_to_statechange


	def create_action_dictionary(self):
		'''
		Create action dictionary: a dict that contains mapping of action index to action name
		'''
		idx = 0 # action idx
		dictionary = {} # action idx --> action name
		dictionary[idx] = "parse_input"
		idx += 1
		dictionary[idx] = "parse_goal"
		idx += 1
		dictionary[idx] = "next_stack"
		idx += 1
		dictionary[idx] = "previous_stack"
		idx += 1
		dictionary[idx] = "next_table"
		idx += 1
		dictionary[idx] = "previous_table"
		idx += 1
		dictionary[idx] = "remove"
		idx += 1
		dictionary[idx] = "add"
		idx += 1
		return dictionary



def test_simulator(verbose=False, use_expert=False):
	self.verbose=verbose
	self.state = self.create_state_representation()
	print('initial state\n', self.state)
	total_reward = 0
	if not use_expert:
		for t in range(self.max_steps+10):
			action_idx = random.choice(list(range(0, self.num_actions)))
			# print('current state\n', self.state)
			self.state, reward, done, truncated, info = self.step(action_idx)
			total_reward += reward
			print('t={}, reward={}, action_idx={}, action={}, done={}, next state\n{}'.format(t, reward, action_idx, self.action_dict[action_idx], done, self.state))
	else:
		expert_actions = self.expert_demo()
		for t in range(len(expert_actions)):
			action_idx = expert_actions[t]
			self.state, reward, done, truncated, info = self.step(action_idx)
			# print('current state\n', self.state)
			total_reward += reward
			print('t={}, reward={}, action_idx={}, action={}, done={}, next state\n{}'.format(t, reward, action_idx, self.action_dict[action_idx], done, self.state))
	print('total_reward', total_reward)
		


class EnvWrapper(dm_env.Environment):
	'''
	Wraps a Simulator object to be compatible with dm_env.Environment
	Reference: 
		https://github.com/wcarvalho/human-sf/blob/da0c65d04be708199ffe48d5f5118b295bfd43a3/lib/dm_env_wrappers.py#L15
		https://github.com/google-deepmind/dm_env/
		https://github.com/google-deepmind/acme/
	'''
	def __init__(self, environment: Simulator, random_number_generator):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		obs_space = self._environment.state
		act_space = self._environment.n_actions-1 # maximum action index
		self._observation_spec = _convert_to_spec(obs_space, name='observation')
		self._action_spec = _convert_to_spec(act_space, name='action')
		self.rng = random_number_generator


	def reset(self) -> dm_env.TimeStep:
		self._reset_next_step = False
		# self.rng = np.random.default_rng(self.rng.integers(low=0, high=100)) # refresh the rng
		self.rng = np.random.default_rng(random.randint(0,100)) # refresh the rng
		observation, info = self._environment.reset(random_number_generator=self.rng)
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
		# return specs.BoundedArray(
		# 	shape=(),
		# 	dtype=dtype,
		# 	minimum=min_val,
		# 	maximum=max_val,
		# 	name=name
		# )
		return specs.DiscreteArray(
			num_values=max_val+1,
			name=name
		)
	elif isinstance(space, np.ndarray):
		min_val, max_val = space.min(), space.max()
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
		rng = np.random.default_rng(1)
		num_blocks, input_stacks, goal_stacks = create_random_problem(difficulty=7, random_number_generator=rng)
		# input_stacks = [[0,1,2]]
		# goal_stacks = [[2,0,1]]
		print('difficulty', num_blocks, '\ninput stacks', input_stacks, '\ngoal stacks', goal_stacks)
		environment = Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks)
		return EnvWrapper(environment, rng)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()


if __name__ == '__main__':
	# random.seed(3)	
	# num_blocks, input_stacks, goal_stacks = create_random_problem(difficulty=7)
	# print('difficulty', num_blocks, '\ninput stacks', input_stacks, '\ngoal stacks', goal_stacks)
	# environment = Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks)
	# environment.test(verbose=True, use_expert=True)
	# print('difficulty', num_blocks, '\ninput stacks', input_stacks, '\ngoal stacks', goal_stacks)
	# print('max_episode_reward', environment.max_episode_reward)
	

	absltest.main()
	
	# rng = np.random.default_rng(1)
	# num_blocks, input_stacks, goal_stacks = create_random_problem(difficulty=7, random_number_generator=rng)
	# simulator = Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks)
	# env = EnvWrapper(simulator, rng)
	# for _ in range(5):
	# 	env.reset()
