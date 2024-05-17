configurations = {
'dataset_path': "/n/home04/yichenli/human-sf/envs/gscan/groundedSCAN/data/dummy_full/dataset.txt", 
'split': "train", # split of the dataset
'save_directory': "/n/home04/yichenli/human-sf/envs/gscan/output/", # output dir
'grid_width': 4,
'grid_height': 4,
'goal_template': ['action', 'color', 'size', 'shape', 'manner'],
'history_length': 10,

'all_actions': ["turn left", "turn right", "walk", "run", "jump", "push", "pull", "stay"],

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

configurations['num_actions'] = len(configurations['all_actions'])
configurations['num_colors'] = len(configurations['colors'])
configurations['num_sizes'] = len(configurations['sizes'])
configurations['num_size_descriptions'] = len(configurations['size_descriptions'])
configurations['num_manners'] = len(configurations['manners'])
configurations['num_transverbs'] = len(configurations['transverbs'])
configurations['num_intransverbs'] = len(configurations['intransverbs'])
configurations['num_directions'] = len(configurations['directions'])