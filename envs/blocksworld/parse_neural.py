'''
learning parse 1 stack of blocks into the brain
'''
import numpy as np
import random
import pprint

from envs.blocksworld.cfg import configurations
from envs.blocksworld import utils
import envs.blocksworld.parse as parse
import envs.blocksworld.AC.bw_apps as bw_apps
import envs.blocksworld.AC.blocks_brain as blocks_brain

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree

cfg = configurations['parse']

class Simulator(parse.Simulator):
	def __init__(self, 
				max_steps = cfg['max_steps'],
				action_cost = cfg['action_cost'],
				reward_decay_factor = cfg['reward_decay_factor'],
				prefix="G",
				verbose=False):
		super().__init__(max_steps = max_steps,
						action_cost = action_cost,
						reward_decay_factor = reward_decay_factor,
						verbose = verbose)
		self.prefix = prefix
		print(f"all_areas: {self.all_areas}")


	def reset(self, shuffle=True, difficulty_mode='curriculum', cur_curriculum_level=0):
		'''
		Return: state: (numpy array with float32), info: (any=None)
		'''
		self.state, info = super().reset(shuffle=shuffle, difficulty_mode=difficulty_mode, cur_curriculum_level=cur_curriculum_level)
		oa = [self.head] + self.node_areas
		self.blocksbrain = blocks_brain.BlocksBrain(blocks_number=self.stack_max_blocks*2, other_areas=oa, 
													p=0.1, eak=10, nean=100, neak=10, db=0.2)
		return self.state.copy(), info

		
	def step(self, action_idx):
		'''
		Return: state, reward, terminated, truncated, info
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
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: # BAD, fiber is already disinhibited/inhibited
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.just_projected = False
			if action_name=="disinhibit_fiber":
				self.blocksbrain.disinhibit_fiber(area1=area1, area2=area2)
				self.blocksbrain.disinhibit_areas(area_names=[area1, area2])
			if action_name=="inhibit_fiber":
				self.blocksbrain.inhibit_fiber(area1=area1, area2=area2)
		elif action_name == "project_star": # state_change_tuple = ([],[]) 
			if [*self.last_active_assembly.values()].count(-1)==len(self.last_active_assembly): # BAD, no active assembly exists
				reward -= self.action_cost 
			elif self.just_projected: # BAD, consecutive project
				reward -= self.action_cost
			else: # GOOD, valid project
				self.assembly_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self.state, self.assembly_dict, self.stateidx_to_fibername, self.last_active_assembly, self.num_assemblies, self.verbose, blocks_area=self.blocks_area)
				for area_name in self.last_active_assembly.keys():  # update state for each area
					if (self.skip_relocated and area_name==self.relocated_area) or (area_name==self.blocks_area):
						continue # only node and head areas need to update
					# update last active assembly in state 
					assert self.area_status[0] == 'last_activated', f"idx 0 in self.area_status {self.area_status} should be last_activated"
					self.state[area_to_stateidx[area_name][self.area_status[0]]] = self.last_active_assembly[area_name] 
					# update the number of self.blocks_area related assemblies in this area
					assert self.area_status[1] == 'num_block_assemblies', f"idx 1 in self.area_status {self.area_status} should be num_block_assemblies"
					count = 0 
					for assembly_info in self.assembly_dict[area_name]:
						connected_areas, connected_assemblies = assembly_info[0], assembly_info[1]
						if self.blocks_area in connected_areas:
							count += 1
					self.state[area_to_stateidx[area_name][self.area_status[1]]] = count
					# update the number of total assemblies in this area
					assert self.area_status[2] == 'num_total_assemblies', f"idx 2 in self.area_status {self.area_status} should be num_total_assemblies"
					self.state[area_to_stateidx[area_name][self.area_status[2]]] = len(self.assembly_dict[area_name])
				# readout stack	and compute reward
				synthetic_readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, len(self.goal), self.blocks_area)
				neural_readout = bw_apps.readout(blocks_brain=self.blocksbrain, stacks_number=1, stacks_lengths=[self.stack_max_blocks], top_areas=[0], prefix=self.prefix)[0]
				print(f"synthetic_readout: {synthetic_readout}, neural_readout: {neural_readout}")
				readout = neural_readout
				units, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.reward_decay_factor)
				reward += self.unit_reward * units
				# update current stack in state
				for ib, sidx in enumerate(area_to_stateidx["current_stack"]):
					self.state[sidx] = readout[ib] if readout[ib] != None else -1
				# update top area
				synthetic_top_area, synthetic_topa, synthetic_topbid = utils.top(self.assembly_dict, self.last_active_assembly, self.head, self.blocks_area)
				neural_top_area, neural_topbid = bw_apps.top(blocks_brain=self.blocksbrain, stack_index=0, prefix=self.prefix)
				print(f"synthetic_top_area {synthetic_top_area}, synthetic_topa {synthetic_topa}, synthetic_topbid {synthetic_topbid}")
				print(f"neural_top_area {neural_top_area}, neural_topbid {neural_topbid}")
				top_area = neural_top_area
				topa = synthetic_topa
				topbid = neural_topbid
				if top_area==None:
					assert readout[0]==None, f"top area is {top_area} but readout is nonempty {readout}"
					self.state[area_to_stateidx["top_area"]] = -1
					self.state[area_to_stateidx["top_assembly"]] = -1
					self.state[area_to_stateidx["top_block"]] = -1
				else:
					self.state[area_to_stateidx["top_area"]] = self.node_areas.index(top_area)
					self.state[area_to_stateidx["top_assembly"]] = topa
					self.state[area_to_stateidx["top_block"]] = topbid
				# update is last block
				synthetic_is_last_block = utils.is_last_block(self.assembly_dict, self.head, top_area, topa, self.blocks_area)
				neural_is_last_block = bw_apps.is_last_block(blocks_brain=self.blocksbrain, stack_index=0, node_index=top_area, block=topbid, prefix=prefix)
				print(f"is_last_block synthetic {synthetic_is_last_block}, neural {neural_is_last_block}")
				is_last_block = neural_is_last_block
				self.state[area_to_stateidx["is_last_block"]] = 1 if is_last_block else 0
			self.just_projected = True
		elif action_name == "activate_block":
			bidx = int(self.state[state_change_tuple[0]]) # currently activated block id
			newbidx = int(bidx) + state_change_tuple[1] # the new block id to be activated (prev -1 or next +1)
			if newbidx < 0 or newbidx >= self.puzzle_max_blocks: # BAD, new block id is out of range
				reward -= self.action_cost
			else: # GOOD, valid activate
				self.state[state_change_tuple[0]] = newbidx # update block id in state vec
				self.last_active_assembly[self.blocks_area] = newbidx # update the last active assembly
				self.blocksbrain.activate_block(index=newbidx)
			self.just_projected = False
		elif action_name == "silence_head":
			if self.last_active_assembly[self.head]== -1: # BAD, head is already silence
				assert self.state[area_to_stateidx[self.head]['last_activated']] == -1, \
				f"in state vector, the last activated assembly in head ({self.state[area_to_stateidx[self.head][0]]}) should already be -1 for repeative silence"
				reward -= self.action_cost
			else: # GOOD, valid silence
				self.last_active_assembly[self.head] = -1 # deactivate head
				self.blocksbrain.inhibit_area(area_name=self.head)
				synthetic_top_area, synthetic_topa, synthetic_topbid = utils.top(self.assembly_dict, self.last_active_assembly, self.head, self.blocks_area)
				neural_top_area, neural_topbid = bw_apps.top(blocks_brain=self.blocksbrain, stack_index=0, prefix=self.prefix)
				print(f"synthetic_top_area {synthetic_top_area}, synthetic_topa {synthetic_topa}, synthetic_topbid {synthetic_topbid}")
				print(f"neural_top_area {neural_top_area}, neural_topbid {neural_topbid}")
				assert synthetic_top_area==None and synthetic_topa==None and synthetic_topbid==None, \
					f"synthetic top area {synthetic_toparea}, topa {synthetic_topa}, and topbid {synthetic_topbid} should all be None after silencing head"
				assert not utils.is_last_block(self.assembly_dict, self.head, synthetic_top_area, synthetic_topa, self.blocks_area), \
					f"synthetic is_last_block should be False after silencing head"
				synthetic_readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, len(self.goal), self.blocks_area)
				assert synthetic_readout == [None]*self.stack_max_blocks, f"synthetic readout {readout} should all be None after silencing head" 
				neural_readout = bw_apps.readout(blocks_brain=self.blocksbrain, stacks_number=1, stacks_lengths=[self.stack_max_blocks], top_areas=[0], prefix=self.prefix)[0]
				print(f"synthetic_readout: {synthetic_readout}, neural_readout: {neural_readout}")
				readout = neural_readout
				units, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.reward_decay_factor)
				reward += units * self.unit_reward
				for sidx, sval in zip(state_change_tuple[0], state_change_tuple[1]):
					self.state[sidx] = sval # update top area, top a, top bid, is last block, current readout
			self.just_projected = False
		else:
			raise ValueError(f"\tError: action_idx {action_idx} is not recognized!")
		self.current_time += 1 # increment step in the episode 
		if self.current_time >= self.max_steps:
			truncated = True
		terminated = self.all_correct and utils.all_fiber_closed(self.state, self.stateidx_to_fibername)
		return self.state.copy(), reward, terminated, truncated, info



def test_simulator(expert=True, repeat=10, verbose=False):
	import time
	sim = Simulator(verbose=verbose)
	pprint.pprint(sim.action_dict)
	start_time = time.time()
	avg_expert_len = []
	for num_blocks in range(sim.stack_max_blocks+1):
		expert_len = []
		print(f"num_blocks {num_blocks}")
		for r in range(repeat):
			state, _ = sim.reset(shuffle=True, difficulty_mode='curriculum', cur_curriculum_level=min(num_blocks+1, 7))
			# state, _ = sim.reset(shuffle=True, difficulty_mode=num_blocks) # specify num of blocks
			print(f'\n\n------------ repeat {r}, state after reset\t{state}') if verbose else 0
			expert_demo = utils.expert_demo_parse(sim.goal, sim.num_blocks) if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			print(f"expert demo {expert_demo}") if verbose else 0
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated},\n\tjust_projected={sim.just_projected}, all_correct={sim.all_correct}, correct_record={sim.correct_record}') if verbose else 0
				print(f'\tnext state {next_state}\t') if verbose else 0
				if terminated:
					break
			readout = utils.synthetic_readout(sim.assembly_dict, sim.last_active_assembly, sim.head, len(sim.goal), sim.blocks_area)
			print(f'end of episode (difficulty={difficulty}), num_blocks={sim.num_blocks}, synthetic readout {readout}, goal {sim.goal}, total reward={rtotal}, time lapse={time.time()-start_time}') if verbose else 0
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
	def __init__(self, environment: Simulator):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		self._environment.reset()
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
		sim = Simulator(stack_max_blocks=7)
		return EnvWrapper(sim)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()


if __name__ == "__main__":

	random.seed(1)
	# test_simulator(expert=False, repeat=500, verbose=False)
	test_simulator(expert=True, repeat=1, verbose=False)
	
	# absltest.main()

