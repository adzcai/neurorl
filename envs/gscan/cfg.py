configurations = {
'dataset_path': "/n/home04/yichenli/human-sf/envs/gscan/groundedSCAN/data/dummy_full/dataset.txt", 
'split': "train", # split of the dataset
'save_directory': "/n/home04/yichenli/human-sf/envs/gscan/output/", # output dir
'max_complexity': 8, # max num of words in a valid sentence sample
'num_fibers': 23, # number of fibers in the brain, will be filled once env is created
'num_areas': 14, # number of areas in the brain, will be filled once env is created

'episode_max_reward': 1, # max reward for solving the entire episode correctly
'max_steps': 300, # maximum number of actions allowed in each episode
'action_cost': 5e-4, # cost for performing any action
'empty_unit': 0.005, # reward unit to give for each correct empty block
'max_assemblies': 100, # maximum number of assemblies for each area in the state representation
'area_status': ['last_activated', 'num_lex_assemblies', 'num_total_assemblies'], # area attributes to encode in state
'num_actions': None, # number of actions, will be filled once env is created

'curriculum': 2, # current curriculum
'compositional': True, # whether to holdout particular language sentences
'compositional_eval': False, # whether in eval mode (if yes, only sampling compositional holdout)
'compositional_holdout': [

], # sentence structures to holdout 
}