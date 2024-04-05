configurations = {
'skip_relocated': True,
'stack_max_blocks': 7, # max num of blocks in any stack
'puzzle_max_blocks': 10, # max num blocks in a planning puzzle (across all stacks)
'episode_max_reward': 1, # max reward for solving the entire episode correctly

'plan':
	{
	'cfg': 'plan',
	'puzzle_max_stacks': 2 , # maximum number of stacks allowed in a puzzle
	'max_steps': 100 , # maximum number of actions allowed in each episode
	'base_block_reward': 3 , # base reward for getting 1 block correct, cumulates with more blocks correct
	'reward_decay_factor': 1.2 , # discount factor for subsequently correct blocks, the larger the faster the decay
	'action_cost': 1e-3 , # cost for performing any action
	'curriculum': 2 , # current difficulty level to focus on in curriculum training. None for fixed curriculum, 2 for dynamic curriculum
	'episode_reward_threshold': 1.4 , # current upper episode reward threshold to proceed to next curriculum level
	'threshold1': 1.4 , # upper episode reward threshold to proceed from curriculum 2 to 3
	'threshold2': 2.0  , # upper episode reward threshold to proceed from curriculum 3 to 4
	'threshold3': 2.3  , # upper episode reward threshold to proceed from curriculum 4 to above, and final stabalizing
	'episode_reward_lowerbound': -1.4 , # reasonable lower bound for determining whether to switch or continue final stabalizing
	'episode_reward_lowerbound_factor': 1.04 , # lower bound for episode reward scales with level
	'up_pressure': 0 , # current pressure counter for determining when to increment curriculum
	'down_pressure': 0 , # current pressure counter for determinng when to decrement curriculum
	'up_pressure_threshold': 1 , # pressure threshold for proceeding to the next curriculum
	'down_pressure_threshold': 1 , # pressure threshold for falling back to the previous curriculum
	},

'parse':
	{
	'cfg': 'parse',
	'max_steps': 200, # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.9, # reward discount factor, descending (first index is most rewarding) if 0 < factor < 1, ascending if factor > 1
	'action_cost': 1e-3, # cost for performing any action
	'max_assemblies': 50, # maximum number of assemblies for each area in the state representation
	'num_fibers': None, # number of fibers in the brain, will be filled once env is created
	'num_areas': None, # number of areas in the brain, will be filled once env is created
	'num_actions': None, # number of actions, will be filled once env is created
	'area_status': ['last_activated', 'num_block_assemblies', 'num_total_assemblies'], # area attributes to encode in state
	},

'remove':
	{
	'cfg': 'remove',
	'max_steps': 100, # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.9, # discount factor for subsequently correct blocks, the larger the faster the decay
	'action_cost': 1e-3, # cost for performing any action
	},

'add':
	{
	'cfg': 'add',
	'max_steps': 100, # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.9, # discount factor for subsequently correct blocks, the larger the faster the decay
	'action_cost': 1e-3, # cost for performing any action
	},

}