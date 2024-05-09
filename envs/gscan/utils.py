from envs.gscan.groundedSCAN.GroundedScan.dataset import GroundedScan
from envs.gscan.cfg import configurations

import random
random.seed(0)

# Load the dataset.
dataset_path = configurations['dataset_path']
dataset = GroundedScan.load_dataset_from_file(dataset_path, save_directory="output", k=1)
all_actions = configurations['all_actions']


# Go over the eaxmples in the training set.
for i, example in enumerate(dataset.get_raw_examples(split="train")):
	if i>=1:
		break
	# Initialize the example in the dataset to obtain the initial state.
	goal, state = dataset.initialize_rl_example(example)
	# Take actions from some policy (NB: here undefined) to interact with the environment.
	total_reward = 0
	done = False
	print(f"\n\ninitial state\n{state} shape {state.shape}")
	# while not done:
	for t in range(1):
		action = random.choice(all_actions)
		state, reward, done = dataset.take_step(action)
		total_reward += reward
		print(f"\nt={t}, action={action}, reward={reward}, state:\n{state}")

def extract_obj_loc(placed_obj_dict):
	return {('irow', 'icol'): 'objid'}

'''
salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

'''