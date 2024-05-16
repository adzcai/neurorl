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
		self.grid_width = configurations['grid_width']
		self.grid_height = configurations['grid_height']
		self.history_length = configurations['history_length']

	def encode_initial_world(self, goal_command, grid_state):
		self.initialize_grid_areas() # empty areas
		self.initialize_history_areas() # empty areas
		self.initialize_goal_areas() # empty areas
		formatted_goal_command = self.format_goal_command(goal_command)
		self.grid_status = self.encode_grid_status(grid_state) # dict to track grid
		self.agent_status = {
							'position': (int(world['agent_position']['row']),int(world['agent_position']['column'])),
							'direction': world['agent_direction'],
							'action': None, # current action 
							'carried_obj': None, # obj with push or pull
							}
		self.encode_goal_areas(formatted_goal_command)
		self.encode_grid_areas(self.grid_status)
		
	def encode_grid_status(self, grid_state):
		self.grid_status = {}
		for irow in self.grid_height:
			for jcol in self.grid_width:
				vec = grid_state[irow, jcol]
				self.grid_status[(irow, jcol)] = {}
				self.grid_status[(irow, jcol)]['hasobj'] = np.any(np.array(vec[:-5])>0)
				self.grid_status[(irow, jcol)]['size'] = np.argmax(vec[:self.num_sizes]) or -1
				self.grid_status[(irow, jcol)]['color'] = np.argmax(vec[self.num_sizes:self.num_sizes+self.num_colors])
				self.grid_status[(irow, jcol)]['shape'] = np.argmax(vec[self.num_sizes+self.num_colors:self.num_sizes+self.num_colors+self.num_shapes])
				if vec[-5]==1: # has agent
					self.agent_status['position'] = (irow, jcol)
					self.agent_status['direction'] = np.argmax(vec[-4:])

	def initialize_goal_areas(self):
		self.goal_representation = [-1] * len(self.goal_template)

	def initialize_history_areas(self):
		self.direction_history = Queue(maxlen=self.history_length, fill=-1)
		self.action_history = Queue(maxlen=self.history_length, fill=-1)
		assert self.direction_history.is_full() and self.action_history.is_full()

	def initialize_grid_areas(self):
		self.grid_assembly_dict = {} # assembly connections
		self.grid_active_assembly = {} # currently activated assembly
		for irow in range(self.grid_height): # area of each grid position
			for jcol in range(self.grid_width):
				self.grid_assembly_dict[(irow, jcol)] = []
				self.grid_active_assembly[(irow, icol)] = -1
		self.grid_assembly_dict['color'] = [[[],[]] for _ in range(self.num_colors)] 
		self.grid_active_assembly['color'] = -1
		self.grid_assembly_dict['shape'] = [[[],[]] for _ in range(self.num_shapes)] 
		self.grid_active_assembly['shape'] = -1
		self.grid_assembly_dict['size'] = [[[],[]] for _ in range(self.num_sizes)] 
		self.grid_active_assembly['size'] = -1
		self.grid_assembly_dict['direction'] = [[[],[]] for _ in range(self.num_directions)] 
		self.grid_active_assembly['direction'] = -1
		self.grid_assembly_dict['action'] = [[[],[]] for _ in range(self.num_actions)] 
		self.grid_active_assembly['action'] = -1

	def extract_obj_loc(self, placed_obj_dict):
		return {('irow', 'icol'): 'objid'}

	def initialize_goal(self):
		goal_repr = []
		for i, struct in enumerate(self.derivation_structure):
			rep = []
			if struct=='verb':
				rep = [0] * nactions
			elif struct=='color':
				rep = [0] * ncolors
			elif struct=='size':
				rep = [0] * nsizes
			elif struct=='shape':
				rep = [0] * nshapes
			elif struct=='adverb':
				rep = [0] * nadverbs
			rep[formatted_goal_command[i]] = 1
			goal_repr.append(rep)
		return goal_repr

	def __initialize_grid_status(self, placed_obj_dict):
		statusdict = {}
		where_are_objects = self.extract_obj_loc(placed_obj_dict)
		for irow in range(self.grid_height):
			for icol in range(self.grid_width):
				if (irow, icol) in where_are_objects.keys():
					statusdict[(irow, icol)] = {where_are_objects[(irow, icol)]}
				else:
					statusdict[(irow, icol)] = None

	def reset(self, dataset_path=configurations['dataset_path'], split=configurations['split'], save_directory=configurations['save_directory']):
		self.dataset = GroundedScan.load_dataset_from_file(self.dataset_path, save_directory=self.save_directory, k=1)
		example = dataset.get_raw_examples(split=split)
		state = dataset.initialize_rl_example(example)
		info = None
		return state, None
	
	def create_state_representation(self):
		return

	def close(self):
		del self.dataset, self.dataset_path
		del self.grid_height, self.grid_width
		del self.grid_assembly_dict, self.grid_active_assembly
		return 

	def create_action_dictionary(self, all_actions):
		# map action idx to action name
		action_dict = {} 
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
