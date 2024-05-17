'''
env for parse language

salloc -p gpu_test -t 0-03:00 --mem=80000 --gres=gpu:1

salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

python envs/language/langenv.py

'''


import numpy as np
import random
import pprint

from envs.gscan.cfg import configurations

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree

from envs.gscan.groundedSCAN.GroundedScan.dataset import GroundedScan


class Queue:
	def __init__(self, maxlen, fill=None):
		self.items = []
		self.maxlen = maxlen
		if fill!=None:
			for _ in range(self.maxlen):
				self.items.append(fill)
	def size(self):
		return len(self.items)
	def put(self, val):
		assert self.size()<=self.maxlen, f"{self.size()} should be <= {self.maxlen}"
		if self.size()==self.maxlen:
			self.items.pop(0) # remove the oldest item
		self.items.append(val)
	def get(self):
		if self.items:
			return self.items.pop(0)
		else:
			return None
	def is_empty(self):
		return self.items == []
	def is_full(self):
		return self.size()==self.maxlen
	def tolist(self):
		return self.items

class Simulator():
	def __init__(self, 
				dataset_path = configurations['dataset_path'],
				save_directory = configurations['save_directory'],
				goal_template = configurations['goal_template'],
				verbose=False):
		self.verbose = verbose
		self.dataset_path = dataset_path
		self.save_directory = save_directory
		self.goal_template = goal_template
		self.area_status = configurations['area_status'], # area attributes to encode in state
		self.action_dict = self.create_action_dictionary(configurations['all_actions'])
		self.num_actions = len(self.action_dict)
		self.colors = configurations['color']
		self.num_colors = len(self.colors)
		self.shapes = configurations['shapes']
		self.num_shapes = len(self.shapes)
		self.sizes = configurations['sizes']
		self.num_sizes = len(self.sizes)
		self.size_descriptions = configurations['size_descriptions']
		self.num_size_descriptions = len(self.size_descriptions)
		self.manners = configurations['manners']
		self.num_manners = len(self.manners)
		self.transverbs = configurations['transverbs']
		self.num_transverbs = len(self.transverbs)
		self.intransverbs = configurations['intransverbs']
		self.num_intransverbs = len(self.intransverbs)
		self.directions = configurations['directions']
		self.num_directions = len(self.directions)
		assert self.num_directions==configurations['num_directions']
		self.grid_width = configurations['grid_width']
		self.grid_height = configurations['grid_height']
		self.history_length = configurations['history_length']

	def reset(self, dataset_path=configurations['dataset_path'], split=configurations['split'], save_directory=configurations['save_directory']):
		self.dataset = GroundedScan.load_dataset_from_file(self.dataset_path, save_directory=self.save_directory, k=1)
		example = dataset.get_raw_examples(split=split)
		raw_goal_command, raw_grid = dataset.initialize_rl_example(example)
		state = self.encode_initial_world(raw_goal_command, raw_grid)
		info = None
		return state, None

	def encode_initial_world(self, raw_goal_command, raw_grid):
		self.formatted_goal = self.format_goal(raw_goal_command)
		self.goal_representation = self.encode_goal_representation(self.formatted_goal)
		self.action_history, self.direction_history = self.initialize_history_representation()
		self.grid_info = self.encode_raw_grid(raw_grid) # dict to track grid		
		self.grid_assembly_dict, self.grid_active_assembly = self.encode_grid_representation(self.grid_info)
		self.state = self.assembly_to_state(
										self.grid_assembly_dict, 
										self.grid_active_assembly,
										self.action_history, self.direction_history,
										self.goal_representation,
										)
		return self.state

	def assembly_to_state(self, grid_assembly_dict, grid_active_assembly, ):
		self.grid_assembly_dict
		self.grid_active_assembly
		self.
		
	def format_goal(self, raw_goal_command):
		formatted = {}
		for gstruct in self.goal_template: 
			formatted[gstruct] = None
		for word in raw_goal_command:
			if word in self.transverbs:
				wid = self.transverb.index(word)
				formatted['action'] = (word, wid)
			elif word in self.intransverbs:
				wid = self.intransverb.index(word)
				formatted['action'] = (word, wid)
			elif word in self.colors:
				wid = self.colors.index(word)
				formatted['color'] = (word, wid)
			elif word in self.sizes:
				wid = self.sizes.index(word)
				formatted['size'] = (word, wid)
			elif word in self.shapes:
				wid = self.shapes.index(word)
				formatted['shape'] = (word, wid)
			elif word in self.manners:
				wid = self.manners.index(word)
				formatted['manner'] = (word, wid)
			else:
				print(f"omitting goal word {word}")
		return formatted

	def encode_goal_representation(self, formatted_goal): 
		# record assembly id for the goal elements
		self.goal_representation = [-1] * len(self.goal_template)
		for i, gstruct in enumerate(self.goal_template):
			if formatted_goal[gstruct]!=None:
				(word, wid) = formatted_goal[gstruct]
				self.goal_representation[i] = wid
		return self.goal_representation

	def initialize_history_representation(self):
		# FIFO queue for the last active assembly id
		self.direction_history = Queue(maxlen=self.history_length, fill=-1)
		self.action_history = Queue(maxlen=self.history_length, fill=-1)
		assert self.direction_history.is_full() and self.action_history.is_full()
		return self.direction_history, self.action_history

	def encode_raw_grid(self, raw_grid):
		self.grid_info = {}
		for irow in self.grid_height:
			for jcol in self.grid_width:
				vec = raw_grid[irow, jcol]
				self.grid_info[(irow, jcol)] = {}
				self.grid_info[(irow, jcol)]['hasobj'] = np.any(np.array(vec[:-5])>0)
				self.grid_info[(irow, jcol)]['size'] = np.argmax(vec[:self.num_sizes]) or -1
				self.grid_info[(irow, jcol)]['color'] = np.argmax(vec[self.num_sizes:self.num_sizes+self.num_colors])
				self.grid_info[(irow, jcol)]['shape'] = np.argmax(vec[self.num_sizes+self.num_colors:self.num_sizes+self.num_colors+self.num_shapes])
				if vec[-5]==1: # has agent
					self.grid_info['agent_position'] = (irow, jcol)
					self.grid_info['agent_direction'] = np.argmax(vec[-4:])
					# self.grid_info['agent_action'] = None # current action 
					# self.grid_info['agent_carried_obj']: None, # obj with push or pull
		return self.grid_info

			
	def encode_grid_representation(self, grid_info):
		self.grid_assembly_dict = {} # assembly dict for grid
		self.grid_active_assembly = {} # currently activated assembly
		for irow in range(self.grid_height): # each grid position
			for jcol in range(self.grid_width): # init empty
				self.grid_assembly_dict[(irow, jcol)] = []
				self.grid_active_assembly[(irow, icol)] = -1
		# init long term areas
		self.grid_assembly_dict['color'] = [[[],[]] for _ in range(self.num_colors)] 
		self.grid_active_assembly['color'] = -1
		self.grid_assembly_dict['shape'] = [[[],[]] for _ in range(self.num_shapes)] 
		self.grid_active_assembly['shape'] = -1
		self.grid_assembly_dict['size'] = [[[],[]] for _ in range(self.num_sizes)] 
		self.grid_active_assembly['size'] = -1
		self.grid_assembly_dict['agent_direction'] = [[[],[]] for _ in range(self.num_directions)] 
		self.grid_active_assembly['agent_direction'] = -1
		# self.grid_assembly_dict['agent_action'] = [[[],[]] for _ in range(self.num_actions)] 
		# self.grid_active_assembly['agent_action'] = -1
		# encode grid info
		for irow in self.grid_height:
			for jcol in self.grid_width:
				thisgrid = grid_info[(irow, jcol)]
				if thisgrid['hasobj']: 
					color = thisgrid['color']
					shape = thisgrid['shape']
					size = thisgrid['size']
					self.grid_assembly_dict[(irow, jcol)].append([['color', 'shape', 'size'], [color, shape, size]])
					self.grid_active_assembly[(irow, jcol)] = 0
					self.grid_assembly_dict['color'][color][0].append((irow, jcol))
					self.grid_assembly_dict['color'][color][1].append(0)
					self.grid_assembly_dict['shape'][shape][0].append((irow, jcol))
					self.grid_assembly_dict['shape'][shape][1].append(0)
					self.grid_assembly_dict['size'][size][0].append((irow, jcol))
					self.grid_assembly_dict['size'][size][1].append(0)
				if grid_info['agent_position']==(irow, jcol): # if agent is here
					direction = grid_info['agent_direction']
					self.grid_active_assembly['agent_direction'] = direction
					if self.grid_assembly_dict[(irow, jcol)] != []: # merge with obj assembly
						assbid = len(self.grid_assembly_dict[(irow, jcol)])-1
						self.grid_assembly_dict[(irow, jcol)][assbid][0].append('agent_direction')
						self.grid_assembly_dict[(irow, jcol)][assbid][0].append(direction)
						self.grid_active_assembly[(irow, jcol)] = assbdict
						self.grid_assembly_dict['agent_direction'][direction][0].append((irow, jcol))
						self.grid_assembly_dict['agent_direction'][direction][1].append(assbdict)
					else: # create new assembly
						self.grid_assembly_dict[(irow, jcol)].append([['agent_direction'], [direction]])
						self.grid_active_assembly[(irow, jcol)] = 0
						self.grid_assembly_dict['agent_direction'][direction][0].append((irow, jcol))
						self.grid_assembly_dict['agent_direction'][direction][1].append(0)

	
	def create_state_representation(self):
		return

	def close(self):
		del self.dataset, self.dataset_path
		del self.grid_height, self.grid_width
		del self.grid_assembly_dict, self.grid_active_assembly
		return 

	def create_action_dictionary(self, all_actions):
		action_dict = {} # map action idx to action name
		for i, a in enumerate(all_actions):
			action_dict[i] = a
		return action_dict


def test_simulator(expert=True, repeat=1, verbose=False):
	import time
	sim = Simulator(verbose=False)
	pprint.pprint(sim.action_dict)
	start_time = time.time()
	avg_expert_len = []
	for complexity in range(2, sim.max_complexity+1):
		expert_len = []
		print(f"\n\ncomplexity: {complexity}")
		for r in range(repeat):
			state, _ = sim.reset(difficulty_mode=complexity) # specify complexity
			print(f'------------ repeat {r}, state after reset\t{state}') if verbose else 0
			expert_demo = utils.expert_demo(sim) if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			print(f"expert demo {expert_demo}") if verbose else 0
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated},\tall_correct={sim.all_correct}, correct_record={sim.correct_record}') if verbose else 0
				# print(f'\tnext state {next_state}\t') if verbose else 0
			readout = utils.synthetic_readout(sim)
			print(f'end of episode (complexity={complexity}), num_words={sim.num_words}, \
					\nsynthetic readout {readout}\n\t{utils.translate(readout)}, \
					\ngoal {sim.goal}\n\t{utils.translate(sim.goal)}, \
					\ntotal reward={rtotal}, time lapse={time.time()-start_time}') if verbose else 0
			if expert:
				assert readout == sim.goal, f"readout {readout} and goal {sim.goal} should be the same"
				assert terminated, "episode should be done"
				theoretical_reward = sim.episode_max_reward - sim.action_cost*nsteps
				assert np.isclose(rtotal, theoretical_reward, 0.05), \
						f"rtotal {rtotal} and theoretical total {theoretical_reward} should be roughly the same"
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


		# Go over the eaxmples in the training set.
		for i, example in enumerate(dataset.get_raw_examples(split="train")):
			# Initialize the example in the dataset to obtain the initial state.
			state = dataset.initialize_rl_example(example)
			# Take actions from some policy (NB: here undefined) to interact with the environment.
			total_reward = 0
			done = False
			while not done:
				action = policy.step(state)
				state, reward, done = dataset.take_step(action)
				total_reward += reward


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
		min_val, max_val = space.min(), configurations['max_assemblies']
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
			for key, value in space.items()		}
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



'''
salloc -p gpu_test -t 0-01:00 --mem=8000 --gres=gpu:1

salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

python envs/language/langenv.py
'''

if __name__ == "__main__":

	random.seed(8)

	test_simulator(expert=False, repeat=500, verbose=False)
	test_simulator(expert=True, repeat=500, verbose=False)
	
	absltest.main()	
