'''
TODO
	systematic holdout test samples for plan (vary by puzzle_num_blocks)
	integrate plan with parse/add/remove agents
	test integrated plan agent on holdout samples
	correct utils.is_last_block() and utils.top()
	check why need wipe when all blocks are removed
DONE
	load trained agent checkpoint
	adjusted plan reward for empty blocks
	add reset: alternate btw add and remove
	add: curriculum
	remove reset: alternate btw add and remove
	remove: curriculum
	parse: curriculum
	better encoder for parse, add, remove
	trainer curriculum for parse, add, remove
	train parse
	train remove
	train add
'''

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
	'reward_decay_factor': 0.99 ,  # reward discount factor, descending (first index is most rewarding) if 0 < factor < 1, ascending if factor > 1
	'sparse_reward': True, # whether to only grant reward when episode terminates. If False, grant intermediate subrewards.
	'action_cost': 1e-3 , # cost for performing any action
	'empty_block_unit': 0.005, # reward unit to give for each correct empty block
	'num_actions': None, # number of actions in the brain, will be filled once env is created
	'action_dict': None, # action dict, will be filled once env is created
	'curriculum': 4, # starting level, determine number of blocks in puzzle, in {0 (uniform), 2,..., puzzle_max_blocks}
	'leak': False, # whether to leak harder puzzles during curriculum
	'compositional': False, # whether in compositional mode
	'compositional_type': 'newblock', # {None, 'newblock', 'newconfig'}, if None: do not apply compositional training, 
															# if 'newblock': holdout a few block ids during training
															# if 'newconfig': holdout a few stack configs during training
	'compositional_holdout': [2,3,5,7], # the list of block ids or stack config to holdout
	},

'parse':
	{
	'cfg': 'parse',
	'max_steps': 200, # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.99, # reward discount factor, descending (first index is most rewarding) if 0 < factor < 1, ascending if factor > 1
	'action_cost': 1e-3, # cost for performing any action
	'max_assemblies': 50, # maximum number of assemblies for each area in the state representation
	'area_status': ['last_activated', 'num_block_assemblies', 'num_total_assemblies'], # area attributes to encode in state
	'num_fibers': None, # number of fibers in the brain, will be filled once env is created
	'num_areas': None, # number of areas in the brain, will be filled once env is created
	'num_actions': None, # number of actions, will be filled once env is created
	'action_dict': None, # action dict, will be filled once env is created
	'cur_curriculum_level': 1, # starting level, determine number of blocks, in {0 (uniform), 1, ..., stack_max_blocks}
	},

'remove':
	{
	'cfg': 'remove',
	'max_steps': 50, # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.95, # discount factor for subsequently correct blocks, the larger the faster the decay
	'action_cost': 1e-3, # cost for performing any action
	'maxnprocesses': 14, # number of add/remove before returning add goal
	'cur_curriculum_level': 0, # starting level, determine current num of processes, in {-1 (uniform), 0, 1, ..., maxnprocesses}
	},

'add':
	{
	'cfg': 'add',
	'max_steps': 100, # maximum number of actions allowed in each episode
	'reward_decay_factor': 0.99, # discount factor for subsequently correct blocks, the larger the faster the decay
	'action_cost': 1e-3, # cost for performing any action
	'maxnprocesses': 10, # number of add/remove before returning add goal
	'cur_curriculum_level': 1, # {-1, 1,2,...,stack_max_blocks} starting level, determine current num of processes, in {-1 (uniform), 0, 1, ..., maxnprocesses}
	},

}