'''
env for parse language

// interactive session
salloc -p gpu_test -t 0-01:00 --mem=8000 --gres=gpu:1
module load python/3.10.12-fasrc01
mamba activate neurorl
'''
import numpy as np
import random
import pprint

import envs.language.utils as utils
from envs.language.cfg import configurations

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree


class Simulator():
	# brain only represent 1 stack
	def __init__(self, 
				max_lexicon = configurations['max_lexicon'], # max num blocks in planning puzzle
				episode_max_reward = configurations['episode_max_reward'],
				max_steps = configurations['max_steps'],
				max_input_length = configurations['max_input_length'],
				action_cost = configurations['action_cost'],
				reward_decay_factor = configurations['reward_decay_factor'],
				area_status = configurations['area_status'],
				max_complexity = configurations['max_complexity'],
				verbose=False):
		self.all_areas, self.lexicon_area, self.det_area, self.all_fibers = utils.init_simulator_areas()
		self.max_lexicon = max_lexicon
		self.max_steps = max_steps # max steps allowed in episode
		self.max_input_length = max_input_length # max num of words in sentence
		self.action_cost = action_cost
		self.reward_decay_factor = reward_decay_factor
		self.episode_max_reward = episode_max_reward
		self.verbose = verbose
		self.area_status = area_status # area attributes to encode in state, default ['last_activated', 'num_lex_assemblies', 'num_total_assemblies']
		self.max_complexity = max_complexity
		self.num_areas = len(self.all_areas)
		self.action_dict = self.create_action_dictionary() 
		self.num_actions = len(self.action_dict)
		self.num_assemblies = self.max_lexicon # total number of assemblies ever created


	def reset(self, difficulty_mode='uniform', cur_curriculum_level=None):
		'''
		Reset environment for new episode.
		Return:
			state: (numpy array with float32)
			info: (any=None)
		'''
		self.num_words, self.goal = utils.sample_episode(difficulty_mode=difficulty_mode, 
														cur_curriculum_level=cur_curriculum_level, 
														max_input_length=self.max_input_length,
														max_complexity=self.max_complexity)
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.num_fibers = len(self.stateidx_to_fibername.keys())
		self.just_projected = False # record if the previous action was project
		self.all_correct = False # if the most recent readout has everything correct
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are ever correct in the episode
		self.current_time = 0 # current step in the episode
		self.num_assemblies = self.max_lexicon
		info = None
		return self.state.copy(), info

	def close(self):
		'''
		Close and clear the environment.
		Return nothing.
		'''
		del self.num_words, self.goal
		del self.unit_reward
		del self.state
		del self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly
		del self.correct_record
		del self.all_correct, self.just_projected
		del self.current_time, self.num_assemblies, self.num_fibers
		return 
		
	def step(self, action_idx):
		'''
		Return: 
			state: (numpy array with float32)
			reward: (float)
			terminated: (boolean)
			truncated: (boolean)
			info: (any=None)
		'''
		action_tuple = self.action_dict[int(action_idx)] # (action name, *kwargs)
		action_name = action_tuple[0]
		state_change_tuple = self.action_to_statechange[int(action_idx)] # (state index, new state value)
		stateidx_to_fibername = self.stateidx_to_fibername # {state vec idx: (area1, area2)} 
		area_to_stateidx = self.area_to_stateidx # {area_name: state vec starting idx}
		reward = -self.action_cost # default cost for performing any action
		terminated = False # whether the episode ended
		truncated = False # end due to max steps
		info = None
		if (action_name == "disinhibit_fiber") or (action_name == "inhibit_fiber"):
			area1, area2 = action_tuple[1], action_tuple[2]
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: 
				reward -= self.action_cost # BAD, fiber is already disinhibited/inhibited
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.just_projected = False
		elif action_name == "project_star": # state_change_tuple = ([],[]) 
			if [*self.last_active_assembly.values()].count(-1)==len(self.last_active_assembly):
				reward -= self.action_cost # BAD, no active assembly exists
			elif self.just_projected: 
				reward -= self.action_cost # BAD, consecutive project
			else: # GOOD, valid project
				self.assembly_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self)
				for area_name in self.last_active_assembly.keys():  # update state for each area
					if area_name==self.lexicon_area:
						continue 
					# update last active assembly in state 
					assert self.area_status[0] == 'last_activated', f"idx 0 in self.area_status {self.area_status} should be last_activated"
					self.state[area_to_stateidx[area_name][self.area_status[0]]] = self.last_active_assembly[area_name] 
					# update the number of self.lexicon_area related assemblies in this area
					assert self.area_status[1] == 'num_lex_assemblies', f"idx 1 in self.area_status {self.area_status} should be num_lex_assemblies"
					count = 0 
					for assembly_info in self.assembly_dict[area_name]:
						connected_areas, connected_assemblies = assembly_info[0], assembly_info[1]
						if self.lexicon_area in connected_areas:
							count += 1
					self.state[area_to_stateidx[area_name][self.area_status[1]]] = count
					# update the number of total assemblies in this area
					assert self.area_status[2] == 'num_total_assemblies', f"idx 2 in self.area_status {self.area_status} should be num_total_assemblies"
					self.state[area_to_stateidx[area_name][self.area_status[2]]] = len(self.assembly_dict[area_name])
				# readout stack	and compute reward
				readout = utils.synthetic_readout(self)
				units, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.reward_decay_factor)
				reward += self.unit_reward * units
				# update current stack in state
				for ib, sidx in enumerate(area_to_stateidx["readout"]):
					self.state[sidx] = readout[ib] if readout[ib] != None else -1
			self.just_projected = True
		elif action_name == "activate_lex":
			lexid = int(self.state[state_change_tuple[0]]) # currently activated lexicon id
			newlexid = int(lexid) + state_change_tuple[1] # the new block id to be activated (prev -1 or next +1)
			if newlexid < 0 or newlexid >= self.max_lexicon:
				reward -= self.action_cost  # BAD, new block id is out of range
			else: # GOOD, valid activate
				self.state[state_change_tuple[0]] = newlexid # update block id in state vec
				self.last_active_assembly[self.lexicon_area] = newlexid # update the last active assembly
			self.just_projected = False
		elif action_name == "clear_det": 
			if self.last_active_assembly[self.det_area]== -1: # BAD, head is already silence
				assert self.state[area_to_stateidx[self.det_area]['last_activated']] == -1, \
				f"in state vector, the last activated assembly in head ({self.state[area_to_stateidx[self.head][0]]}) should already be -1 for repeative silence"
				reward -= self.action_cost
			else: # GOOD, valid clear
				self.last_active_assembly[self.det_area] = -1 # deactivate det
				readout = utils.synthetic_readout(self)
				units, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.reward_decay_factor)
				reward += units * self.unit_reward
				for sidx, sval in zip(state_change_tuple[0], state_change_tuple[1]):
					self.state[sidx] = sval # update state
			self.just_projected = False
		else:
			raise ValueError(f"\tError: action_idx {action_idx} is not recognized!")
		self.current_time += 1 # increment step in the episode 
		if self.current_time >= self.max_steps:
			truncated = True
		terminated = self.all_correct and utils.all_fiber_closed(self.state, self.stateidx_to_fibername)
		return self.state.copy(), reward, terminated, truncated, info

	def create_state_representation(self):
		'''
		Initialize the episode state in the environmenmt. 
		Return:
			state: (numpy array with float32)
				state representation
				[cur lex readout (initialized as all -1s),
				input lex (padding -1 at the end),
				input syntactic types (padding -1 at the end),
				fiber inhibition status (initialized as all closed 0s),
				last activated assembly idx in the area (initialized as all -1s), 
				number of blocks-connected assemblies in each area (initialized as 0s, or max_lexicon for lexicon area),
				number of all assemblies in each area (initialized as 0s, or max_lexicon for BLOCKS area),
				]
			action_to_statechange: (dict)
				map action index to change in state vector
				{action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
			area_to_stateidx: (dict)
				map area name to indices in state vector
				{area: [corresponding indices in state vector]}
			stateidx_to_fibername: (dict)
				mapping state vector index to fiber between two areas
				{state idx: (area1, area2)}
			assembly_dict: (dict)
				dictionary storing assembly associations that currently exist in the brain
				{area: [assembly_idx0[source areas[A0, A1], source assembly_idx[a0, a1]], 
						assembly_idx1[[A3], [a3]], 
						assembly_idx2[[A4], [a4]], 
						...]}
				i.e. area has assembly_idx0, which is associated/projected from area A0 assembly a0, and area A1 assembly a1
			last_active_assembly: (dict)
				dictionary storing the latest activated assembly idx in each area
				{area: assembly_idx}
				assembly_idx = -1 means that no previously activated assembly exists
		'''
		state_vec = [] # state vector
		action_to_statechange = {} # action -> state change, {action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
		action_idx = 0 # action index, the order should match that in self.action_dict
		state_vector_idx = 0 # initialize the idx in state vec to be changed
		area_to_stateidx = {} # dict of index of each area
		stateidx_to_fibername = {} # mapping of state vec index to fiber name
		assembly_dict = {} # {area: [assembly1 associations...]}
		last_active_assembly = {} # {area: assembly_idx}
		# encode current readout sentence
		area_to_stateidx["readout"] = []
		for _ in range(self.max_input_length):
			state_vec.append(-1) # empty word
			area_to_stateidx["readout"].append(state_vector_idx) # state idx for readout
			state_vector_idx += 1 # increment state index
		# encode goal sentence words
		area_to_stateidx["goalword"] = []
		for ib in range(self.max_input_length):
			if self.goal[0][ib]==None:  # filler for empty block
				state_vec.append(-1)
			else:
				state_vec.append(self.goal[0][ib])
			area_to_stateidx["goalword"].append(state_vector_idx) # state idx for goal
			state_vector_idx += 1 # increment state index
		# encode goal sentence word types
		area_to_stateidx["goaltype"] = []
		for ib in range(self.max_input_length):
			if self.goal[1][ib]==None:  # filler for nontype
				state_vec.append(-1)
			else:
				state_vec.append(self.goal[1][ib])
			area_to_stateidx["goaltype"].append(state_vector_idx) # state idx for goal
			state_vector_idx += 1 # increment state index
		# encode fiber inhibition status
		for (area1, area2) in self.all_fibers:
			# add fiber status to state vector
			state_vec.append(0) # fiber should be locked upon initialization
			stateidx_to_fibername[state_vector_idx] = (area1, area2) # record state idx -> fiber name
			# action -> state change: disinhibit fiber
			assert self.action_dict[action_idx][0]=="disinhibit_fiber" and self.action_dict[action_idx][1]==area1 and self.action_dict[action_idx][2]==area2, \
					f"action_index {action_idx} should have (disinhibit_fiber, {area1}, {area2}), but action_dict has {self.action_dict}"
			action_to_statechange[action_idx] = ([state_vector_idx], 1) # open fiber
			action_idx += 1
			# action -> state change: inhibit fiber
			assert self.action_dict[action_idx][0]=="inhibit_fiber" and self.action_dict[action_idx][1]==area1 and self.action_dict[action_idx][2]==area2, \
					f"action_index {action_idx} should have (inhibit_fiber, {area1}, {area2}), but action_dict has {self.action_dict}"
			action_to_statechange[action_idx] = ([state_vector_idx], 0) # close fiber
			action_idx += 1
			# increment state idx
			state_vector_idx += 1
		for area in self.all_areas:
			last_active_assembly[area] = -1 # initialize area with no activated assembly
			assembly_dict[area] = [] # will become {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
		# # encode area inhibition status
		# for area in self.all_areas:
		# 	if area==self.lexicon_area:
		# 		continue
		# 	state_vec.append(0) # area locked initially
		# 	assert self.action_dict[action_idx][0]=="disinhibit_area" and self.action_dict[action_idx][1]==area, \
		# 				f"action_index {action_idx} should have (disinhibit_area, {area}, None), but action_dict has {self.action_dict}"
		# 	action_to_statechange[action_idx] = ([state_vector_idx], 1) # open area
		# 	action_idx += 1	
		# 	assert self.action_dict[action_idx][0]=="inhibit_area" and self.action_dict[action_idx][1]==area, \
		# 				f"action_index {action_idx} should have (inhibit_area, {area}, None), but action_dict has {self.action_dict}"
		# 	action_to_statechange[action_idx] = ([state_vector_idx], 0) # close area
		# 	action_idx += 1
		# 	state_vector_idx += 1
		# action -> state change: strong project (i.e. project star)
		assert self.action_dict[action_idx][0]=="project_star", \
			f"action_index {action_idx} should have (project_star, None), but action_dict has {self.action_dict}"
		action_to_statechange[action_idx] = ([],[]) # no pre-specified new state values, things will be updated after project
		action_idx += 1
		# encode area info
		for istatus, status_name in enumerate(self.area_status): 
			for area_name in self.all_areas: 
				if istatus==0: # encode last activated assembly index in this area
					area_to_stateidx[area_name] = {status_name: state_vector_idx} # area -> state idx
					state_vec.append(-1) # initialize most recent assembly as none 
				elif istatus==1: # encode number of lexicon-connected assemblies in this area
					area_to_stateidx[area_name][status_name] = state_vector_idx # area -> state idx
					if area_name==self.lexicon_area:
						state_vec.append(self.max_lexicon)
					else:
						state_vec.append(0)
				elif istatus==2: # encode number of total assemblies in this area
					area_to_stateidx[area_name][status_name] = state_vector_idx # area -> state idx
					if area_name==self.lexicon_area:
						state_vec.append(self.max_lexicon)
					else:
						state_vec.append(0)
				else:
					raise ValueError(f"there are only {len(self.area_status)} status for each area in state, but requesting {istatus}")
				state_vector_idx += 1 # increment state vector idx
		# action -> state change: activate next block assembly
		assert self.action_dict[action_idx][0]=="activate_lex" and self.action_dict[action_idx][1]=="next", \
			f"action_index {action_idx} should have (activate_lex, next), but action_dict has {self.action_dict}"
		action_to_statechange[action_idx] = (area_to_stateidx[self.lexicon_area][self.area_status[0]], +1) 
		action_idx += 1
		assert self.action_dict[action_idx][0]=="activate_lex" and self.action_dict[action_idx][1]=="prev", \
			f"action_index {action_idx} should have (activate_lex, prev), but action_dict has {self.action_dict[action_idx]}"
		# action -> state change: activate previous block assembly
		action_to_statechange[action_idx] = (area_to_stateidx[self.lexicon_area][self.area_status[0]], -1) 
		action_idx += 1
		# action -> state change: clear det
		action_to_statechange[action_idx] = ([area_to_stateidx[self.det_area][self.area_status[0]], \
												area_to_stateidx[self.det_area][self.area_status[1]],
												area_to_stateidx[self.det_area][self.area_status[2]]],
											[-1, 0, 0])
		# initialize assembly dict for lex area, other areas will be updated during project
		assembly_dict[self.lexicon_area] = [[[],[]] for _ in range(self.max_lexicon)] 
		return np.array(state_vec, dtype=np.float32), \
				action_to_statechange, \
				area_to_stateidx, \
				stateidx_to_fibername, \
				assembly_dict, \
				last_active_assembly


	def create_action_dictionary(self):
		'''
		Create action dictionary: a dict that contains mapping of action index to action name
		Assuming action bundle: disinhibit_fiber entails disinhibiting the two areas and the fiber, opening the fibers in both directions
		Return:
			dictonary: (dict) 
				{action_idx : (action name, *kwargs)}
		'''
		idx = 0 # action idx
		dictionary = {} # action idx --> (action name, *kwargs)
		# disinhibit and inhibit fibers
		for (area1, area2) in self.all_fibers:
			dictionary[idx] = ("disinhibit_fiber", area1, area2)
			idx += 1
			dictionary[idx] = ("inhibit_fiber", area1, area2)
			idx += 1
		# for area in self.all_areas:
		# 	if area == self.lexicon_area:
		# 		continue
		# 	dictionary[idx] = ("disinhibit_area", area, None)
		# 	idx += 1
		# 	dictionary[idx] = ("inhibit_area", area, None)
		# 	idx += 1
		# project star
		dictionary[idx] = ("project_star", None, None)
		idx += 1
		# activate lex
		dictionary[idx] = ("activate_lex", 'next', None)
		idx += 1
		dictionary[idx] = ("activate_lex", 'prev', None)
		idx += 1
		# clear det
		dictionary[idx] = ("clear_det", None, None)
		idx += 1
		return dictionary



def test_simulator(expert=True, repeat=1, verbose=False):
	import time
	sim = Simulator(verbose=verbose)
	pprint.pprint(sim.action_dict)
	start_time = time.time()
	avg_expert_len = []
	for complexity in range(1, sim.max_complexity+1):
		expert_len = []
		print(f"complexity: {complexity}")
		for r in range(repeat):
			# state, _ = sim.reset(difficulty_mode='curriculum', cur_curriculum_level=min(complexity+1, stack_max_blocks))
			state, _ = sim.reset(difficulty_mode=complexity) # specify num of blocks
			print(f'\n\n------------ repeat {r}, state after reset\t{state}') if verbose else 0
			expert_demo = utils.expert_demo_language(sim) if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			print(f"expert demo {expert_demo}") if verbose else 0
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated},\n\tjust_projected={sim.just_projected}, all_correct={sim.all_correct}, correct_record={sim.correct_record}') if verbose else 0
				print(f'\tnext state {next_state}\t') if verbose else 0
			readout = utils.synthetic_readout(sim)
			print(f'end of episode (complexity={complexity}), num_words={sim.num_words}, synthetic readout {readout}, goal {sim.goal}, total reward={rtotal}, time lapse={time.time()-start_time}') if verbose else 0
			if expert:
				assert readout == sim.goal, f"readout {readout} and goal {sim.goal} should be the same"
				assert terminated, "episode should be done"
				assert np.isclose(rtotal, sim.episode_max_reward-sim.action_cost*nsteps), \
						f"rtotal {rtotal} and theoretical total {sim.episode_max_reward-sim.action_cost*nsteps} should be roughly the same"
				expert_len.append(len(expert_demo))
		avg_expert_len.append(np.mean(expert_len)) if expert else 0
	print(f"\n\navg expert demo length {avg_expert_len}\n\n")





if __name__ == "__main__":

	random.seed(1)
	test_simulator(expert=True, repeat=1, verbose=True)
	# test_simulator(expert=True, repeat=200, verbose=False)
	
