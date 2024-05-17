configurations = {
'max_complexity': 5,#8, # max num of words in a valid sentence sample
'num_fibers': 13,#23, # number of fibers in the brain, will be filled once env is created
'num_areas': 10,#14, # number of areas in the brain, will be filled once env is created
'max_sentence_length': 5,#11, # maximum sentence length (depending on the output structure), will be filled once env created
'num_pos': 4,#7, # total number of possible part of speech
'num_words': 20,#27, # total number of possible volcabularies 

'episode_max_reward': 1, # max reward for solving the entire episode correctly
'max_steps': 300, # maximum number of actions allowed in each episode
'action_cost': 5e-4, # cost for performing any action
'empty_unit': 0.005, # reward unit to give for each correct empty block
'max_assemblies': 50, # maximum number of assemblies for each area in the state representation
'area_status': ['last_activated', 'num_lex_assemblies', 'num_total_assemblies'], # area attributes to encode in state
'num_actions': None, # number of actions, will be filled once env is created
'action_dict': None, # action dictionary, will be filled once env is created

'curriculum': 2, # current curriculum
'spacing': False, # whether to fill -1s to the goal pos structure and goal lex structure
'compositional': True, # whether to holdout particular language sentences
'compositional_eval': False, # whether in eval mode (if yes, only sampling compositional holdout)
'compositional_holdout': 
[
['det','noun', 'intransverb', -1,-1,],
[-1,'noun', 'transverb', 'det','noun', ],
],



# [
# [-1,'adj','noun', 'intransverb', -1,-1,-1, -1,-1,-1, 'adv'],
						
# ['det',-1,'noun', 'transverb', -1,'adj','noun', -1,-1,-1,-1], 
# [-1,-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
# [-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', -1],

# ['det',-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
# [-1,'adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
# [-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
# [-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],

# ['det','adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
# ['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
# [-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
# [-1,-1,'noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
# ['det','adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],

# ['det','adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
# ['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun',-1],
# [-1,'adj','noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
# [-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun','adv'],
# [-1,-1,'noun', 'transverb', 'det','adj','noun', 'prep','det','noun',-1],
# [-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun','adv'],

# ], # sentence structures to holdout 

}