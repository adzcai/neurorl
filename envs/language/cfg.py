configurations = {
'max_complexity': 4, # max num of words in a valid sentence sample
'episode_max_reward': 1, # max reward for solving the entire episode correctly

'max_steps': 200, # maximum number of actions allowed in each episode
'action_cost': 1e-3, # cost for performing any action
'empty_unit': 0.005, # reward unit to give for each correct empty block
'max_assemblies': 50, # maximum number of assemblies for each area in the state representation
'num_fibers': None, # number of fibers in the brain, will be filled once env is created
'num_areas': None, # number of areas in the brain, will be filled once env is created
'num_actions': None, # number of actions, will be filled once env is created
'area_status': ['last_activated', 'num_lex_assemblies', 'num_total_assemblies'], # area attributes to encode in state

}