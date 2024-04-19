configurations = {
'max_input_length': 3, # max num of words in input sentence
'max_lexicon': 20, # max num different words in the dictionary
'max_complexity': 3, # max num of words in a valid sentence sample
'episode_max_reward': 1, # max reward for solving the entire episode correctly

'max_steps': 200, # maximum number of actions allowed in each episode
'reward_decay_factor': 0.9, # reward discount factor, descending (first index is most rewarding) if 0 < factor < 1, ascending if factor > 1
'action_cost': 1e-3, # cost for performing any action
'max_assemblies': 50, # maximum number of assemblies for each area in the state representation
'num_fibers': None, # number of fibers in the brain, will be filled once env is created
'num_areas': None, # number of areas in the brain, will be filled once env is created
'num_actions': None, # number of actions, will be filled once env is created
'area_status': ['last_activated', 'num_lex_assemblies', 'num_total_assemblies'], # area attributes to encode in state

}