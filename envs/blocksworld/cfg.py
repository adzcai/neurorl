configurations = {
'skip_relocated': True,
'stack_max_blocks': 7, # max num of blocks in any stack
'puzzle_max_blocks': 10, # max num blocks in a planning puzzle (across all stacks)
'puzzle_max_stacks': 5, # maximum number of stacks allowed in a puzzle
'episode_max_reward': 1, # max reward for solving the entire episode correctly

'plan':
	{
	'cfg': 'plan',
	'max_steps': 200 , # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.9 , # discount factor for subsequently correct blocks, the larger the faster the decay
	'action_cost': 1e-3 , # cost for performing any action
	'curriculum': 0 , # current curriculum level focusing on specific number of blocks. {None, 0, 2,..., puzzle_max_blocks}
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
	'max_steps': 50, # maximum number of actions allowed in each episode
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