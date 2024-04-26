'''
learn to plan in blocksworld using neural actions parse, add, remove
suppose we have 3 segments in the state: current stacks, table stacks, goal stacks
	(along with other info, e.g. pointer indices, correct history, etc.)
'''
import random
import numpy as np
import pprint
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
DONE
	cur stack and goal stack are: top (must have) -> bottom (optional)
	expert demo input and goal
	state vec 1D
	base_block_reward -> unit reward
'''

class Simulator():
	def __init__(self,
			  puzzle_max_stacks=configurations['puzzle_max_stacks'], 
			  puzzle_max_blocks=configurations['puzzle_max_blocks'], 
			  stack_max_blocks=configurations['stack_max_blocks'], 
			  episode_max_reward=configurations['episode_max_reward'],
			  max_steps=cfg['max_steps'],
			  reward_decay_factor=cfg['reward_decay_factor'],
			  sparse_reward=cfg['sparse_reward'],
			  action_cost=cfg['action_cost'],
			  empty_block_unit=cfg['empty_block_unit'],
			  evaluation=False, eval_puzzle_num_blocks=None,
			  compositional=cfg['compositional'], 
			  compositional_type=cfg['compositional_type'], 
			  compositional_holdout=cfg['compositional_holdout'],
			  test_puzzle=None,
			  verbose=False):
		assert cfg['cfg'] == 'plan', f"cfg is {cfg['cfg']}"
		self.puzzle_max_stacks = puzzle_max_stacks
		self.puzzle_max_blocks = puzzle_max_blocks
		self.stack_max_blocks = stack_max_blocks
		self.episode_max_reward = episode_max_reward
		self.reward_decay_factor = reward_decay_factor 
		self.sparse_reward = sparse_reward
		self.action_cost = action_cost
		self.max_steps = max_steps # maximum number of actions allowed in an episode
		self.verbose = verbose
		self.action_dict = self.create_action_dictionary() 
		self.num_actions = len(self.action_dict)
		self.empty_block_unit = empty_block_unit
		self.evaluation = evaluation # whether is in evaluation mode
		self.eval_puzzle_num_blocks = eval_puzzle_num_blocks # eval puzzle lvl
		self.compositional = compositional # whether is compositional setting
		self.compositional_type = compositional_type # {None, 'newblock', 'newconfig'}
		self.compositional_holdout = compositional_holdout # list of block ids to holdout for compositional training
		self.test_puzzle = test_puzzle

	def close(self):
		del self.state
		del self.input_stacks, self.goal_stacks, self.flipped_goal_stacks, self.puzzle_num_blocks
		del self.unit_reward, self.sparse_reward
		del self.action_dict, self.action_to_statechange
		del self.current_time
		del self.evaluation, self.eval_puzzle_num_blocks
		del self.compositional, self.compositional_type, self.compositional_holdout
		return 

	def reset(self, puzzle_num_blocks=None, curriculum=None):
		info = None
		self.puzzle_num_blocks, input_stacks, goal_stacks = None, None, None
		if self.evaluation: # eval mode
			if self.test_puzzle!=None: # a predetermined test puzzle is given
				input_stacks = self.test_puzzle[0]
				goal_stacks = self.test_puzzle[1]
				self.puzzle_num_blocks = sum([1 for stack in input_stacks for b in stack])
			else:
				assert self.eval_puzzle_num_blocks!=None, f"eval_puzzle_num_blocks {self.eval_puzzle_num_blocks} should be given in evaluation mode"
				self.puzzle_num_blocks, input_stacks, goal_stacks = utils.sample_random_puzzle(puzzle_max_stacks=self.puzzle_max_stacks, 
																						puzzle_max_blocks=self.puzzle_max_blocks, 
																						stack_max_blocks=self.stack_max_blocks,
																						puzzle_num_blocks=self.eval_puzzle_num_blocks, 
																						curriculum=None, leak=False,
																						compositional=self.compositional, 
																						compositional_type=self.compositional_type,
																						compositional_eval=self.compositional,
																						compositional_holdout=self.compositional_holdout,
																						) 
		else: # training mode
			if curriculum==None:
				import envs.blocksworld.cfg as config
				curriculum = config.configurations['plan']['curriculum']
			leak = cfg['leak']
			self.puzzle_num_blocks, input_stacks, goal_stacks = utils.sample_random_puzzle(puzzle_max_stacks=self.puzzle_max_stacks, 
																					puzzle_max_blocks=self.puzzle_max_blocks, 
																					stack_max_blocks=self.stack_max_blocks,
																					puzzle_num_blocks=self.eval_puzzle_num_blocks, 
																					curriculum=curriculum, leak=leak,
																					compositional=self.compositional, 
																					compositional_type=self.compositional_type,
																					compositional_eval=False,
																					compositional_holdout=self.compositional_holdout,
																					) 
		assert puzzle_num_blocks==None or (puzzle_num_blocks == self.puzzle_num_blocks)
		# print(f"reset, puzzle_num blocks {self.puzzle_num_blocks}, inputs{input_stacks}, goal{goal_stacks}")
		# format the input and goal to same size: [puzzle_max_stacks, stack_max_blocks]
		self.goal_stacks = [[-1 for _ in range(self.stack_max_blocks)] for _ in range(self.puzzle_max_stacks)] 
		for istack in range(len(goal_stacks)): # stack reads from top/highest block to bottom/lowest block, then filled by -1s
			for jblock in range(len(goal_stacks[istack])): 
				self.goal_stacks[istack][jblock] = goal_stacks[istack][jblock] 
		self.flipped_goal_stacks = self.flip(self.goal_stacks)  # flipped stack reads from bottom/lowest block to top/highest block, then filled by -1s
		self.input_stacks = [[-1 for _ in range(self.stack_max_blocks)] for _ in range(self.puzzle_max_stacks)]
		for istack in range(len(input_stacks)): # stack reads from top/highest block to bottom/lowest block, then filled by -1s
			for jblock in range(len(input_stacks[istack])): 
				self.input_stacks[istack][jblock] = input_stacks[istack][jblock]
		self.unit_reward = self.__calculate_unit_reward()
		self.current_time = 0  
		self.state, self.action_to_statechange = self.create_state_representation() 
		return self.state.copy(), info
	
	def __set_state(self, changes):
		'''
		Set state idxs to new vals
		changes: [[state idxs to be changed], [new values to be filled]]
		'''
		idxs, vals = changes[0], changes[1]
		assert len(idxs)==len(vals), f"idxs {idxs} and vals {vals} should have same size"
		for i, v in zip(idxs, vals):
			self.state[i] = v

	def __add_to_state(self, changes):
		'''
		Add vals to state idxs
		changes: [[state idxs to be changed], [new values to be added]]
		'''
		idxs, vals = changes[0], changes[1]
		assert len(idxs)==len(vals), f"idxs {idxs} and vals {vals} should have same size"
		for i, v in zip(idxs, vals):
			self.state[i] += v

	def __stack_empty(self, istart, length):
		# all empty (-1) blocks in a stack in state
		assert istart>=0 and istart+length<len(self.state)
		return np.all(self.state[istart: istart+length] == -1)

	def __stack_full(self, istart, length):
		# no empty (-1) blocks in a stack in state
		assert istart>=0 and istart+length<len(self.state)
		return np.all(self.state[istart: istart+length] != -1)
	
	def __pop_from_stack(self, istart, length):
		# remove the top block from a stack in the state 
		if length==1:
			b = self.state[istart]
			self.state[istart] = -1
			return b
		b = self.state[istart] # the top block to be popped
		newstack = np.roll(self.state[istart : istart+length], -1) # shift to the left
		newstack[-1] = -1 # fill the last position with empty
		self.state[istart: istart+length] = newstack
		assert self.state[istart+length-1]==-1, f"last item after pop should be -1, but stack is {self.state[istart: istart+length]}"
		return b

	def __insert_top_block(self, block, istart, length):
		# add the top block to a stack in the state 
		if length==1:
			self.state[istart] = block
			return
		newstack = np.roll(self.state[istart: istart+length], 1) # shift everything to the right
		newstack[0] = block # fill the first position with new block
		self.state[istart: istart+length] = newstack
		assert self.state[istart]==block, f"first item after insert should be {block}, but stack is {self.state[istart: istart+length]}"
	
	def __ith_empty_position(self, istart, length, i=1):
		# find the state idx corresponding to the ith empty position 
		for idx in range(istart, istart+length):
			if self.state[idx] == -1.:
				i -= 1
			if i == 0:
				return idx
		assert False, f"table {self.state[istart: istart+length]} (istart {istart}, length {length}) has no empty position!"

	def step(self, action_idx):
		action_idx = int(action_idx)
		action_name = self.action_dict[action_idx] 
		action_to_statechange = self.action_to_statechange
		stack_pointer = int(self.state[action_to_statechange['stack_pointer']])
		assert 0 <= stack_pointer < self.puzzle_max_stacks, f"stack_pointer {stack_pointer} should be 0<=, <{self.puzzle_max_stacks}"
		table_pointer = int(self.state[action_to_statechange['table_pointer']])
		assert 0 <= table_pointer < self.puzzle_max_blocks, f"table_pointer {table_pointer} should be 0<=, <{self.puzzle_max_stacks}"
		input_parsed = int(self.state[action_to_statechange['input_parsed']])
		goal_parsed = int(self.state[action_to_statechange['goal_parsed']])
		assert (input_parsed==0 or input_parsed==1) and (goal_parsed==0 or goal_parsed==1), f"input_parsed {input_parsed}, goal_parsed {goal_parsed}"
		reward = -self.action_cost # default cost for performing any action
		terminated = False # whether the episode ended
		truncated = False # end due to max steps
		info = None
		if (not input_parsed) or (not goal_parsed):
			if (not input_parsed) and (not goal_parsed) and (action_name != "parse_input") and (action_name != "parse_goal"):
				reward -= self.action_cost*2  # BAD, perform any other actions before input or goal is parsed
				print('\tboth input and goal not parsed yet') if self.verbose else 0
			elif (not input_parsed) and action_name == "parse_input": # GOOD, parse input for first time
				self.__set_state(action_to_statechange[action_name])
				assert self.state[action_to_statechange['input_parsed']]==1, f"state {self.state} should have input parsed being 1, but have {self.state[action_to_statechange['input_parsed']]}"
			elif (not goal_parsed) and action_name == "parse_goal": # GOOD, parse goal for first time
				self.__set_state(action_to_statechange[action_name])
				assert self.state[action_to_statechange['goal_parsed']]==1, f"state {self.state} should have input parsed being 1, but have {self.state[action_to_statechange['goal_parsed']]}"
			else: # BAD, perform other actions when one of input or goal still not parsed
				reward -= self.action_cost*2
				print('\teither input or goal not parsed yet') if self.verbose else 0
		elif action_name == "next_stack":
			if stack_pointer + 1 >= self.puzzle_max_stacks: # BAD, new cur pointer idx out of range
				reward -= self.action_cost
				print('\tcur pointer out of range') if self.verbose else 0
			else: # GOOD, next stack
				self.__add_to_state(action_to_statechange[action_name])
		elif action_name == "previous_stack":
			if stack_pointer == 0: # BAD, cur pointer is already minimum
				reward -= self.action_cost
				print('\tcur pointer out of range') if self.verbose else 0
			else: # GOOD, previous stack
				self.__add_to_state(action_to_statechange[action_name])
		elif action_name == "next_table":
			if table_pointer + 1 >= self.puzzle_max_blocks: # BAD, new table pointer out of range
				reward -= self.action_cost
				print('\ttable pointer out of range') if self.verbose else 0
			else: # GOOD, next table stack
				self.__add_to_state(action_to_statechange[action_name])
		elif action_name == "previous_table":
			if table_pointer == 0: # BAD, table pointer is already minimum
				reward -= self.action_cost
				print('\ttable pointer out of range') if self.verbose else 0
			else: # GOOD, previous table stack
				self.__add_to_state(action_to_statechange[action_name])
		elif action_name == "remove":
			if self.__stack_empty(istart=action_to_statechange['cur_stacks_begin'][stack_pointer], length=self.stack_max_blocks): 
				reward -= self.action_cost # BAD, cur stack is empty
				print('\tnothing to remove, cur stack empty') if self.verbose else 0
			else: # GOOD, pop the top block from cur stack
				block = self.__pop_from_stack(istart=action_to_statechange['cur_stacks_begin'][stack_pointer], length=self.stack_max_blocks)
				sidx = self.__ith_empty_position(istart=action_to_statechange['table'][0], length=self.puzzle_max_blocks, i=1)
				self.__set_state([[sidx], [block]])
				print('\tremove top block', block) if self.verbose else 0
				units, terminated, correct_history = self.__reward(cur_stacks=self.flip(self.decode_cur_stacks()), goal_stacks=self.flipped_goal_stacks)
				reward += units * self.unit_reward if (not self.sparse_reward) else 0
				self.__set_state([action_to_statechange['correct_history'], correct_history])
		elif action_name == "add":
			if self.__stack_empty(istart=action_to_statechange['table'][table_pointer], length=1): 
				reward -= self.action_cost # BAD, nothing to add, table stack is empty
				print('\tnothing to add, table stack empty') if self.verbose else 0
			elif self.__stack_full(istart=action_to_statechange['cur_stacks_begin'][stack_pointer], length=self.stack_max_blocks): 
				reward -= self.action_cost # BAD, intent to add to cur stack, but stack full
				print('\tintend to add to full stack, last block in stack is',self.state[stack_pointer+self.stack_max_blocks-1]) if self.verbose else 0
			else: # GOOD, add the block to cur stack
				block = self.__pop_from_stack(istart=action_to_statechange['table'][table_pointer], length=1) # pop from table
				self.__insert_top_block(block=block, istart=action_to_statechange['cur_stacks_begin'][stack_pointer], length=self.stack_max_blocks)
				units, terminated, correct_history = self.__reward(cur_stacks=self.flip(self.decode_cur_stacks()), goal_stacks=self.flipped_goal_stacks)
				reward += units * self.unit_reward if (not self.sparse_reward) else 0
				self.__set_state([action_to_statechange['correct_history'], correct_history])
		elif action_name == "parse_input": # BAD, parse input repetitively
			assert input_parsed
			self.__set_state(action_to_statechange[action_name])
			reward -= self.action_cost*2
			print('\tinput parsed again, reset') if self.verbose else 0
		elif action_name == "parse_goal": # BAD, parse goal repetitively
			assert goal_parsed
			self.__set_state(action_to_statechange[action_name])
			reward -= self.action_cost*2
			print('\tgoal parsed again') if self.verbose else 0
		self.current_time += 1
		if self.current_time >= self.max_steps:
			truncated = True
		if self.sparse_reward and terminated:
			reward += self.episode_max_reward
		return self.state.copy(), reward, terminated, truncated, info

	def flip(self, stacks):
		'''
		Input/goal stacks are given as normally ordered blocks 
		(read from highest/top block to lowest/bottom block, then filled by -1s)
		Return stacks read from lowest block to highest block, then filled by -1s
		'''
		newstacks = []
		for s in stacks:
			news = []
			for ib in range(len(s)-1, -1, -1): # back to front
				if s[ib]==-1: # skip empty positions
					continue
				else:
					news.append(s[ib])
			news += [-1] * (len(s)-len(news)) # fill the rest with -1s
			assert len(news)==self.stack_max_blocks, f"wrong length after flipping stack {s} => {news}"
			newstacks.append(news)
		return newstacks

	def decode_cur_stacks(self):
		# each stack will be ordered as top/highest to bottom/lowest, then filled with -1s
		cur_stacks = []
		for istack in range(self.puzzle_max_stacks): 
			stack = []
			for jblock in range(self.stack_max_blocks):
				stack.append(int(self.state[self.action_to_statechange['cur_stacks_begin'][istack] + jblock]))
			cur_stacks.append(stack)
		return cur_stacks
		
	def __calculate_unit_reward(self):
		'''
		Assuming self.input_stacks and self.goal_stacks already exist.
		Empty blocks will receive reward of 0.1 units, no decay applied.
		'''
		total_units = 0
		for istack in range(self.puzzle_max_stacks):
			nblocks = 0
			nempty = 0
			for jblock in range(self.stack_max_blocks):
				if self.goal_stacks[istack][jblock] != -1:
					nblocks += 1
				else:
					nempty += 1
			assert nblocks + nempty == self.stack_max_blocks, f"nblocks {nblocks} + nempty {nempty} should be {self.stack_max_blocks}"
			total_units += sum([self.reward_decay_factor**i for i in range(nblocks)])
			total_units += self.empty_block_unit * nempty
		unit_value = self.episode_max_reward / total_units
		return unit_value

	def __stack_readout_reward(self, readout, goal, correct_record):
		'''
		Calculate reward by comparing current stack readout with goal stack.
			Reward decays from top to bottom block.
			Reward a block only if all its previous (higher) blocks are correct.
		Input: 
			readout: current stack readout from the brain (chain of blocks from top to bottom)
			goal: the goal stack chain of blocks (from top/high to bottom/low) to be matched
			correct_record: binary record for how many blocks are already correct in this episode (index 0 records the correctness of block idx 0).
				will not get reward anymore if the index was already correct in the episode.
		Return: 
			score: (float)
				units of reward to award. 
				an unit is the smallest amount of reward to give for a correct index.
			all_correct: (boolean)
				whether current readout has all blocks correct
			correct_record: (numpy array with binary values)
				history of correct in the episode.
		'''
		assert len(readout) == len(goal), f"readout length {len(readout)} and goal length {len(goal)} should match."
		units = 0
		num_correct = 0
		prerequisite = [0 for _ in range(len(goal))] # the second block will received reward only if first block is also readout correctly
		for jblock in range(len(goal)): # read from top to bottom
			if readout[jblock] == goal[jblock]: # block match
				prerequisite[jblock] = 1
				num_correct += 1
				if correct_record[jblock]==0 and utils.check_prerequisite(prerequisite, jblock): # reward new correct blocks in this episode
					if readout[jblock]==-1: # empty position gets smaller reward
						units += self.empty_block_unit
					else: # actual block match gets larger reward (with decay)
						units += (self.reward_decay_factor**jblock) # reward scales by position
					correct_record[:jblock+1] = 1 # set the block record and all preceeding record to 1, record history for episode
		all_correct = (num_correct > 0) and (num_correct == len(goal))
		if all_correct:
			assert np.all([r==1 for r in correct_record])
		return units, all_correct, correct_record

	def __reward(self, cur_stacks, goal_stacks):
		# assuming cur_stacks and goal_stacks are already flipped: read from lowest block to highest block, then -1s
		units = 0
		all_correct = []
		correct_history = []
		for i, (cstack, gstack) in enumerate(zip(cur_stacks, goal_stacks)):
			assert len(cstack)==len(gstack), f"cur stack {cstack} and goal stack {gstack} should have same length"
			ncorrect = int(self.state[self.action_to_statechange['correct_history'][i]])
			assert 0 <= ncorrect <= self.stack_max_blocks, f"correct history value {ncorrect} of stack {i} should be 0<=, <={self.stack_max_blocks}"
			crecord = np.zeros_like(cstack)
			crecord[:ncorrect] = 1
			u, ac, crecord = self.__stack_readout_reward(cstack, gstack, crecord)
			units += u
			all_correct.append(ac)
			correct_history.append(max(ncorrect, np.sum(crecord)))
			assert 0<= correct_history[-1] <= self.stack_max_blocks
		return units, all(all_correct), correct_history
		
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
				[cur stack 1 top block, cur stack 1 second block, ..., cur stack 1 bottom block, cur stack 2 top block, ..., cur puzzle_max_stacks bottom block,
				goal stack 1 top block, goal stack 1 second block, ..., goal stack 1 bottom block, goal stack 2 top block, ..., goal puzzle_max_stacks bottom block,
				table stack 1 block, table stack 2 block, ..., table stack puzzle_max_blocks block,
				correct history record (num of correct blocks throughout episode) for stack 1, ..., correct_history record for stack puzzle_max_stacks,
				cur stack pointer (int 0 to puzzle_max_stacks-1), 
				table pointer (int 0 to puzzle_max_blocks-1),
				input ever parsed throughout episode (0 or 1), 
				goal ever parsed throughout episode (0 or 1)]
			action_to_statechange: (dict)
				{action_name: [[state idxs to change], [new values in state]], ...,
				'cur_stacks_begin': [state idxs for the first block in each current stack],
				'table': [state idxs for each table stack],
				'correct_history': [state idxs for the correct history of each stack],
				'stack_pointer': state idx for current stack pointer,
				'table_pointer': state idx for table pointer,}
		'''
		state_vec = []
		state_idx = 0
		action_to_statechange = {}
		# action -> state change: parse input
		action_to_statechange['parse_input'] = [[],[]] # if parse input called, where and what to update in state
		# info -> state idx: each cur stack top block to state location
		action_to_statechange['cur_stacks_begin'] = []
		# encode current stacks
		for istack in range(self.puzzle_max_stacks):
			action_to_statechange['cur_stacks_begin'].append(state_idx)
			for jblock in range(self.stack_max_blocks):
				state_vec.append(-1) # initialized as empty
				action_to_statechange['parse_input'][0].append(state_idx)
				action_to_statechange['parse_input'][1].append(self.input_stacks[istack][jblock])
				state_idx += 1
		assert len(action_to_statechange['cur_stacks_begin']) == self.puzzle_max_stacks
		# action -> state change: parse goal
		action_to_statechange['parse_goal'] = [[], []] # if parse goal called, where and what to update in state
		# encode goal stacks
		for istack in range(self.puzzle_max_stacks):
			for jblock in range(self.stack_max_blocks):
				state_vec.append(-1) # initialized as empty
				action_to_statechange['parse_goal'][0].append(state_idx)
				action_to_statechange['parse_goal'][1].append(self.goal_stacks[istack][jblock])
				state_idx += 1
		# info -> state idx: table block to state location
		action_to_statechange['table'] = [] # will be filled with state idxs for each table stack
		# encode table stacks
		for _ in range(self.puzzle_max_blocks):
			state_vec.append(-1) # initialized as empty
			action_to_statechange['table'].append(state_idx) 
			action_to_statechange['parse_input'][0].append(state_idx) 
			action_to_statechange['parse_input'][1].append(-1) # clear table when parse input
			state_idx += 1
		assert len(action_to_statechange['table'])==self.puzzle_max_blocks
		# info -> state idx: correct_history record of each stack to state location
		action_to_statechange['correct_history'] = []
		# encode correct_history record
		for _ in range(self.puzzle_max_stacks): 
			state_vec.append(0) # num correct blocks in each stack
			action_to_statechange['correct_history'].append(state_idx)
			state_idx += 1
		assert len(action_to_statechange['correct_history'])==self.puzzle_max_stacks
		# action -> state change: next cur stack
		action_to_statechange['next_stack'] = [[], []]
		# encode cur stack pointer
		state_vec.append(0) # initialize to first stack
		action_to_statechange['next_stack'][0].append(state_idx)
		action_to_statechange['next_stack'][1].append(+1) # stack pointer will be incremented by 1
		action_to_statechange['parse_input'][0].append(state_idx) # reset cur pointer when parse input
		action_to_statechange['parse_input'][1].append(0)
		# action -> state change: prev cur stack
		action_to_statechange['previous_stack'] = [[], []]
		action_to_statechange['previous_stack'][0].append(state_idx)
		action_to_statechange['previous_stack'][1].append(-1)
		# info -> state idx: flag for cur stack pointer to state location
		action_to_statechange['stack_pointer'] = state_idx
		state_idx += 1
		# action -> state change: next table stack
		action_to_statechange['next_table'] = [[], []]
		# encode table pointer
		state_vec.append(0) # initialize to first stack
		action_to_statechange['next_table'][0].append(state_idx)
		action_to_statechange['next_table'][1].append(+1)
		action_to_statechange['parse_input'][0].append(state_idx) # reset table pointer when parse input
		action_to_statechange['parse_input'][1].append(0)
		# action -> state change: prev table stack
		action_to_statechange['previous_table'] = [[], []]
		action_to_statechange['previous_table'][0].append(state_idx)
		action_to_statechange['previous_table'][1].append(-1)
		# info -> state idx: flag for table pointer to state location
		action_to_statechange['table_pointer'] = state_idx
		state_idx += 1
		# encode whether input parsed
		state_vec.append(0) # initialize to False
		# info -> state idx: flag for input parsed to state location
		action_to_statechange['input_parsed'] = state_idx
		action_to_statechange['parse_input'][0].append(state_idx) # update flag when parse input
		action_to_statechange['parse_input'][1].append(1)
		state_idx += 1
		# encode whether goal parsed
		state_vec.append(0) # initialize to False
		# info -> state idx: flag for goal parsed to state location
		action_to_statechange['goal_parsed'] = state_idx
		action_to_statechange['parse_goal'][0].append(state_idx) # update flag when parse goal
		action_to_statechange['parse_goal'][1].append(1)
		state_idx += 1
		assert state_idx == len(state_vec)
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



def test_simulator(expert=True, repeat=10, verbose=False):
	sim = Simulator(verbose=verbose)
	pprint.pprint(sim.action_dict)
	avg_expert_len = []
	for puzzle_num_blocks in range(2, sim.puzzle_max_blocks+1):
		expert_len = []
		print(f"puzzle_num_blocks {puzzle_num_blocks}")
		for r in range(repeat):
			state, info = sim.reset(puzzle_num_blocks=None, curriculum=puzzle_num_blocks) # use curriculum distribution
			# state, info = sim.reset(puzzle_num_blocks=puzzle_num_blocks, curriculum=None) # specify num blocks
			print(f'------------ repeat {r}, state after reset\t{state}') if verbose else 0
			expert_demo = utils.expert_demo_plan(sim) if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			print(f"expert_demo: {expert_demo}")  if verbose else 0
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated}') if verbose else 0
				print(f'\tnext state {next_state}\t') if verbose else 0
			readout = sim.decode_cur_stacks()
			print(f'end of episode (puzzle_num_blocks={puzzle_num_blocks}), num_blocks={sim.puzzle_num_blocks}, synthetic readout {readout}, goal {sim.goal_stacks}, total reward={rtotal}')  if verbose else 0
			if expert:
				assert readout == sim.goal_stacks, f"readout {readout} and goal {sim.goal_stacks} should be the same"
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
	def __init__(self, environment: Simulator):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		obs_space = self._environment.state
		act_space = self._environment.num_actions-1 # maximum action index
		self._observation_spec = _convert_to_spec(obs_space, name='observation')
		self._action_spec = _convert_to_spec(act_space, name='action')


	def reset(self) -> dm_env.TimeStep:
		self._reset_next_step = False
		observation, info = self._environment.reset()
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
	elif isinstance(space, np.ndarray): # observation/state
		min_val, max_val = space.min(), configurations['puzzle_max_blocks']
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
		environment = Simulator()
		environment.reset()
		return EnvWrapper(environment)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()


if __name__ == '__main__':
	# random.seed(1)
	# test_simulator(expert=False, repeat=2000, verbose=False)
	test_simulator(expert=True, repeat=1000, verbose=False)

	absltest.main()
