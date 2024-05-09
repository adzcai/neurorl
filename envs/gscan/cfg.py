configurations = {
'dataset_path': "/n/home04/yichenli/human-sf/envs/gscan/groundedSCAN/data/dummy_full/dataset.txt", 
'split': "train", # split of the dataset
'save_directory': "/n/home04/yichenli/human-sf/envs/gscan/output/", # output dir
'grid_width': 4,
'grid_height': 4,
'goal_template': ['action', 'color', 'size', 'shape', 'manner'],
'history_length': 10,

'all_actions': ["turn left", "turn right", "walk", "run", "jump", "push", "pull", "stay"],
'num_actions': 8, 

'shapes': ["circle", "square", "cylinder"],
'colors': ["red", "blue", "green", "yellow"],
'sizes': list(range(1, 5)),
'size_descriptions': ["small", "big"], 
'manners':  ["quickly", "slowly", "while zigzagging", "while spinning", "cautiously", "hesitantly"],
'transverbs': ["push", "pull"],
'intransverbs': ["walk"],
'directions': ["e", "s", "w", "n"],


'num_fibers': None, # will be filled once env is created
'num_areas': None, #will be filled once env is created
'max_assemblies': 50, # maximum number of assemblies for each area in the state representation
'area_status': ['last_activated', 'num_lex_assemblies', 'num_total_assemblies'], # area attributes to encode in state


}