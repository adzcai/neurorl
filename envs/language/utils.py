'''
env for parse language

salloc -p gpu_test -t 0-03:00 --mem=80000 --gres=gpu:1

salloc -p test -t 0-01:00 --mem=200000 

module load python/3.10.12-fasrc01
mamba activate neurorl

python envs/language/langenv.py

'''
import random
import pptree
import copy
import itertools
import pprint 
import numpy as np

# brain areas
LEX = "LEX"
DET = "DET"
SUBJ = "SUBJ"
OBJ = "OBJ"
VERB = "VERB"
PREP = "PREP"
PREP_P = "PREP_P"
ADJ = "ADJ"
ADVERB = "ADVERB"
ROOT = 'ROOT'
SUBJ_P = 'SUBJ_P'
VERB_P = 'VERB_P'
OBJ_P = 'OBJ_P'
NOUN = 'NOUN'
# all areas
# AREAS = [LEX, DET, NOUN, ADJ, PREP, VERB, ADVERB, SUBJ, OBJ, PREP_P, OBJ_P, SUBJ_P, VERB_P, ROOT]
AREAS = [LEX, DET, NOUN, VERB, SUBJ, OBJ, OBJ_P, SUBJ_P, VERB_P, ROOT]
# INHIBITABLE_AREAS = [DET, ADJ, NOUN]
INHIBITABLE_AREAS = [DET, NOUN]
# all lexicon and their part of speech types
LEXEME_DICT = {
		'det': ['the', 'those', 'these'], 
		'noun': ['dogs', 'cats', 'mice', 'people', 'girls', 'bags', 'books', 'glasses'], 
		'transverb': ['chase', 'love', 'bite', 'saw', 'push', 'follow'], 
		# 'prep': ['with', 'over'], 
		# 'adj': ['big', 'sad', 'cute', 'curious'], 
		'intransverb': ['run', 'fly', 'roll'], 
		# 'adv': ['quickly', 'slowly', 'slightly'], 
		}
ALL_WORDS = [w for l in LEXEME_DICT.values() for w in l] # vocabulary
CFG = { # context free grammar for decoding
		ROOT: [SUBJ_P, VERB_P],
		# VERB_P: [VERB, OBJ_P, PREP_P, ADVERB],
		VERB_P: [VERB, OBJ_P],
		SUBJ_P: [SUBJ], 
		OBJ_P: [OBJ], 
		# PREP_P: [PREP, DET, NOUN],
		# SUBJ: [DET, ADJ, NOUN],
		SUBJ: [DET, NOUN],
		# OBJ: [DET, ADJ, NOUN],
		OBJ: [DET, NOUN],
		# following are basetype
		# ADVERB: [LEX], 
		VERB: [LEX],
		NOUN: [LEX],
		DET: [LEX],
		# ADJ: [LEX],
		# PREP: [LEX],
		}
FIBERS = [(a1, a2) for a1 in list(CFG.keys())[::-1] for a2 in CFG[a1]]

def output_format(cfg=CFG):
	outformat = []
	for component in cfg[ROOT]:
		stack = [component]
		while len(stack)>0: # DFS
			basetype = stack.pop(-1)
			if cfg[basetype]!=[LEX]:
				for bt in cfg[basetype][::-1]:
					stack.append(bt)
			else:
				outformat += [basetype]
	return outformat
OUTPUT_FORMAT = output_format(CFG)
print(f"ALL_WORDS ({len(ALL_WORDS)}): {ALL_WORDS}\
		\nAREAS ({len(AREAS)}): {AREAS}\
		\nFIBERS ({len(FIBERS)}): {FIBERS}\
		\nPOS ({len(LEXEME_DICT.keys())}): {LEXEME_DICT.keys()}\
		\nOUTPUT_FORMAT ({len(OUTPUT_FORMAT)}): OUTPUT_FORMAT")

def synthetic_readout(simulator, cfg=CFG, verbose=False):
	assembly_dict = copy.deepcopy(simulator.assembly_dict)
	last_active_assembly = copy.deepcopy(simulator.last_active_assembly)
	decoded = []
	for component in cfg[ROOT]:
		stack = [(component, True)]
		print(f"decoding component {component}") if verbose else 0
		while len(stack)>0: # DFS
			(node, success) = stack.pop() # pop the last one
			print(f"\tlooking at node {node} ({success})") if verbose else 0
			assembly_dict, last_active_assembly, stimulate_successes = stimulate(source=node, 
																			destinations=cfg[node], 
																			assembly_dict=assembly_dict, 
																			last_active_assembly=last_active_assembly)
			if cfg[node]!=[LEX]:
				for n, suc in zip(cfg[node][::-1], stimulate_successes[::-1]):
					stack.append((n, suc and success))
			else:
				assembly_dict, last_active_assembly, haslex = stimulate(source=node, 
																			destinations=cfg[node], 
																			assembly_dict=assembly_dict, 
																			last_active_assembly=last_active_assembly)
				wordid = last_active_assembly[LEX] if (success and haslex[0]) else -1
				decoded += [wordid]
				word = "_" if wordid==-1 else ALL_WORDS[wordid]
				print(f"\t\tbase case! decoeded word {wordid} ({word})") if verbose else 0
	assert len(decoded)==len(OUTPUT_FORMAT)
	print(f"decoded sequence: {decoded}") if verbose else 0
	if simulator.spacing==False: # remove spacing 
		compactdecoded = []
		for d in decoded:
			if d != -1:
				compactdecoded.append(d)
		compactdecoded += [-1] * (simulator.max_sentence_length-len(compactdecoded))
		decoded = compactdecoded
		assert len(decoded)==len(OUTPUT_FORMAT)
	return decoded

def translate(wordids, wordlist=ALL_WORDS):
	# word ids to words
	words = []
	for wid in wordids:
		if wid==-1:
			words.append("")
		else:
			words.append(wordlist[wid])
	return words

def init_simulator_areas():
	return AREAS, INHIBITABLE_AREAS, LEX, FIBERS, ALL_WORDS, OUTPUT_FORMAT

def get_action_idx(atuple, action_dict):
	(aname, arg1, arg2) = atuple
	if (aname, arg1, arg2) in action_dict.values():
		return list(action_dict.keys())[list(action_dict.values()).index((aname, arg1, arg2))]
	elif (aname, arg2, arg1) in action_dict.values():
		return list(action_dict.keys())[list(action_dict.values()).index((aname, arg2, arg1))] 
	else:
		raise ValueError(f"Action idx for aname: {aname}, arg1 {arg1}, arg2 {arg2} do not exist in action_dict\n{action_dict}")

def get_action_idxs(action_tuples, action_dict):
	idxs = []
	for stage in action_tuples:
		for atuple in stage:
			idxs.append(get_action_idx(atuple, action_dict))
	return idxs


def parse_noun(action_dict):
	action_tuples = [
				[
				("disinhibit_fiber", NOUN, LEX),
				("disinhibit_area", NOUN, None),
				],
				
				[("project_star", None, None),],

				[
				("inhibit_fiber", NOUN, LEX), 
				("inhibit_fiber", SUBJ, DET), 
				# ("inhibit_fiber", SUBJ, ADJ), 
				("inhibit_fiber", SUBJ, NOUN), 
				("inhibit_fiber", SUBJ, SUBJ_P), 
				("inhibit_fiber", OBJ, DET), 
				# ("inhibit_fiber", OBJ, ADJ),
				("inhibit_fiber", OBJ, NOUN), 
				("inhibit_fiber", OBJ, OBJ_P), 
				# ("inhibit_fiber", PREP_P, PREP), 
				# ("inhibit_fiber", PREP_P, DET), 
				# ("inhibit_fiber", PREP_P, NOUN), 
				("inhibit_area", NOUN, None),
				# ("inhibit_area", ADJ, None),
				("inhibit_area", DET, None),
				],
			]
	return get_action_idxs(action_tuples, action_dict)

def parse_transverb(action_dict):
	action_tuples = [
				[
				("disinhibit_fiber", ROOT, SUBJ_P),
				("disinhibit_fiber", SUBJ_P, SUBJ),
				("disinhibit_fiber", SUBJ, DET),
				# ("disinhibit_fiber", SUBJ, ADJ),
				("disinhibit_fiber", SUBJ, NOUN),
				("disinhibit_area", DET, None),
				# ("disinhibit_area", ADJ, None),
				("disinhibit_area", NOUN, None),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", ROOT, SUBJ_P),
				("inhibit_fiber", SUBJ_P, SUBJ),
				("inhibit_fiber", SUBJ, DET),
				# ("inhibit_fiber", SUBJ, ADJ),
				("inhibit_fiber", SUBJ, NOUN),
				("inhibit_area", DET, None),
				# ("inhibit_area", ADJ, None),
				("inhibit_area", NOUN, None),
				],

				[
				("disinhibit_fiber", LEX, VERB),
				],

				[("project_star", None, None),],

				[
				("disinhibit_fiber", VERB, VERB_P),
				("disinhibit_fiber", VERB_P, ROOT),
				],

				[("project_star", None, None),],
	
				[
				("inhibit_fiber", LEX, VERB),
				("disinhibit_fiber", VERB_P, OBJ_P),
				("disinhibit_fiber", OBJ_P, OBJ),
				("disinhibit_fiber", OBJ, DET),
				# ("disinhibit_fiber", OBJ, ADJ),
				("disinhibit_fiber", OBJ, NOUN),
				],
			]
	return get_action_idxs(action_tuples, action_dict)

def parse_intransverb(action_dict):
	action_tuples = [
				[
				("disinhibit_fiber", ROOT, SUBJ_P),
				("disinhibit_fiber", SUBJ_P, SUBJ),
				("disinhibit_fiber", SUBJ, DET),
				# ("disinhibit_fiber", SUBJ, ADJ),
				("disinhibit_fiber", SUBJ, NOUN),
				("disinhibit_area", DET, None),
				# ("disinhibit_area", ADJ, None),
				("disinhibit_area", NOUN, None),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", SUBJ_P, SUBJ),
				("inhibit_fiber", SUBJ, DET),
				# ("inhibit_fiber", SUBJ, ADJ),
				("inhibit_fiber", SUBJ, NOUN),
				],

				[
				("disinhibit_fiber", LEX, VERB),
				],

				[("project_star", None, None),],

				[
				("disinhibit_fiber", VERB, VERB_P),
				("disinhibit_fiber", VERB_P, ROOT),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, VERB),
				("inhibit_fiber", VERB_P, OBJ_P),
				("inhibit_fiber", OBJ_P, OBJ),
				("inhibit_area", DET, None),
				# ("inhibit_area", ADJ, None),
				("inhibit_area", NOUN, None),
				],
			]
	return get_action_idxs(action_tuples, action_dict)

def parse_adverb(action_dict):
	action_tuples = [
				[
				("inhibit_fiber", LEX, DET),
				("inhibit_fiber", LEX, ADJ),
				("inhibit_fiber", LEX, NOUN),
				("inhibit_fiber", LEX, VERB),
				("inhibit_fiber", LEX, PREP),
				("disinhibit_fiber", LEX, ADVERB),
				("disinhibit_fiber", ADVERB, VERB_P),
				("disinhibit_fiber", VERB_P, ROOT),
				],

				[("project_star", None, None),],

				[("inhibit_fiber", LEX, ADVERB)], 
			]
	return get_action_idxs(action_tuples, action_dict)

def parse_det(action_dict):
	action_tuples = [
				[
				("disinhibit_area", DET, None),
				("disinhibit_fiber", LEX, DET),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, DET),
				],
			]
	return get_action_idxs(action_tuples, action_dict)

def parse_adj(action_dict):
	action_tuples = [
				[
				# ("inhibit_fiber", DET, PREP_P),
				# ("inhibit_fiber", NOUN, PREP_P),
				("disinhibit_area", ADJ, None),
				("disinhibit_fiber", LEX, ADJ),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, ADJ),
				],
			]
	return get_action_idxs(action_tuples, action_dict)

def parse_prep(action_dict):
	action_tuples = [
				[
				# ("disinhibit_area", PREP, None),
				# ("disinhibit_area", PREP_P, None),
				("disinhibit_fiber", LEX, PREP),
				("disinhibit_fiber", PREP, PREP_P),
				("disinhibit_fiber", PREP_P, VERB_P),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, PREP),
				("disinhibit_fiber", DET, PREP_P),
				("disinhibit_fiber", NOUN, PREP_P),
				],
			]
	return get_action_idxs(action_tuples, action_dict)


def go_activate(curid, newid, action_goto_next=[31], action_goto_prev=[32]):
	'''
	return a list of action idxs to activate item newid, starting from previous activated item curid
	'''
	actions = []
	diff = curid - newid
	while diff != 0:
		if diff > 0:
			actions += action_goto_prev
			curid -= 1
		else:	
			actions += action_goto_next
			curid += 1
		diff = curid - newid
	return actions

def all_fiber_area_closed(simulator):
	# Check whether all fibers and areas in the state vector are closed.
	fibers_closed = np.all([simulator.state[i]==0 for i in simulator.stateidx_to_fibername.keys()])
	areas_closed = np.all([simulator.state[simulator.area_to_stateidx[a]['opened']]==0 for a in simulator.inhibitable_areas])
	return fibers_closed and areas_closed

def calculate_unit_reward(num_valid_items, num_total_items, empty_unit, episode_max_reward=1):
	empty_items_reward = empty_unit * (num_total_items - num_valid_items) # total units for correct empty items 
	valid_items_reward = num_valid_items # total units for correct real items
	unit_reward = episode_max_reward / (empty_items_reward + valid_items_reward)
	return unit_reward

def calculate_readout_reward(readout, goal, correct_record, empty_unit):
	assert len(readout) == len(goal), f"readout length {len(readout)} and goal length {len(goal)} should match."
	units = 0
	num_correct = 0
	for iword in range(len(goal)): 
		if readout[iword] == goal[iword]: # match
			num_correct += 1
			if correct_record[iword]==0: # reward new correct word in this episode
				if goal[iword]==-1: # empty position gets smaller reward
					units += empty_unit
				else: # actual block match gets larger reward (with decay)
					units += 1
				correct_record[iword] = 1 # record history for episode
	all_correct = (num_correct > 0) and (num_correct == len(goal))
	if all_correct:
		assert np.all([r==1 for r in correct_record])
	return units, all_correct, correct_record

def stimulate(source, destinations, assembly_dict, last_active_assembly):
	sourceaid = last_active_assembly[source]
	stimulate_successes = [False]*len(destinations)
	if sourceaid==-1: # source is silent
		return assembly_dict, last_active_assembly, stimulate_successes
	for idest, dest in enumerate(destinations): # stimulate from source to each dest
		# print(f"stimulate source {source}, dest {dest}, assembly_dict[source]: {assembly_dict[source]}, sourceaid {sourceaid}")
		if dest in assembly_dict[source][sourceaid][0]: # if source assembly is connected with dest
			destloc = assembly_dict[source][sourceaid][0].index(dest) # find the connected dest assembly
			destaid = assembly_dict[source][sourceaid][1][destloc]
			last_active_assembly[dest] = destaid # update active assembly in dest
			stimulate_successes[idest] = True
	return assembly_dict, last_active_assembly, stimulate_successes

def sample_sentence(complexity, max_sentence_length, spacing, compositional, compositional_eval, compositional_holdout, verbose=False):
	'''
	Has to be a sentence with at least 1 noun and 1 verb
		[DET, NOUN, VERB, DET, NOUN]
		['det', 'noun', 'transverb' or 'intransverb', 'det', 'noun']
	Return: number of real words in the sentence, word ids, partr of speech ids
	'''	
	structures = [] 
	assert complexity>=2, f"complexity ({complexity}) should be >=2"
	if complexity==2:
		structures = [
					[-1,'noun', 'intransverb', -1,-1,],
					]
	elif complexity==3:
		structures = [
					[-1,'noun', 'transverb', -1,'noun', ],
					['det','noun', 'intransverb', -1,-1,],
					]
	elif complexity==4:
		structures = [
					['det','noun', 'transverb', -1,'noun', ],
					[-1,'noun', 'transverb', 'det','noun', ],
					]
	elif complexity==5:
		structures = [
					['det','noun', 'transverb', 'det','noun',],
					]
	if compositional and compositional_eval:
		assert complexity>2, f"there is no holdout structures to eval in compleixty 2, complexity ({complexity}) should be >2"
	keep_sampling = True
	while keep_sampling:
		struct = random.choice(structures) # choose a random sentence structure
		if not compositional: 
			keep_sampling = False
		elif compositional_eval: # comp and eval
			if struct in compositional_holdout:
				keep_sampling = False
		else: # comp and train
			if struct not in compositional_holdout:
				keep_sampling = False
	words, poss = [], [] # word ids, part of speech ids
	sentence = ""
	for r in struct:
		w = "_, "
		wid=-1
		rid=-1
		if r!=-1:
			w = random.choice(list(LEXEME_DICT[r]))
			wid = ALL_WORDS.index(w)
			rid = list(LEXEME_DICT.keys()).index(r)
			w += ", "
		sentence += w
		words.append(wid)
		poss.append(rid)
	assert len(poss)==len(words), f"len of poss {poss} should match words {words}"
	print(f"sample sentence with complexity {complexity}, struct {struct},\nsentence: {sentence}") if verbose else 0
	nwords = complexity # number of real words
	if len(poss)<max_sentence_length: # pad empty positions at the end, if any
		words += [-1]*(max_sentence_length-len(words))
		poss += [-1]*(max_sentence_length-len(poss))
	if spacing==False: # remove spacing within a sentence structure
		compactwords = []
		compactposs = []
		for i, (word, pos) in enumerate(zip(words, poss)):
			if word!=-1:
				assert pos!=-1, f"the {i}-th word and pos should both be nonempty\n{words}\n{poss}"
				compactwords.append(word)
				compactposs.append(pos)
		compactwords += [-1] * (max_sentence_length-len(compactwords)) # pad the end with empty 
		compactposs += [-1] * (max_sentence_length-len(compactposs))
		assert len(compactwords)==len(compactposs), f"length of compactwords {compactwords} and compactposs {compactposs} should equal"
		words = compactwords
		poss = compactposs
	return nwords, words, poss

def sample_sentence_complex(complexity, max_sentence_length, spacing, compositional, compositional_eval, compositional_holdout, verbose=False):
	'''
	Has to be a sentence with at least 1 noun and 1 verb
		[DET, ADJ, NOUN, VERB, DET, ADJ, NOUN, PREP, DET, NOUN, ADVERB]
		['det', 'adj', 'noun', 'transverb' or 'intransverb', 'det', 'adj', 'noun', 'prep', 'det', 'noun', 'adv']
	Return: number of real words in the sentence, word ids, partr of speech ids
	'''	
	structures = [] 
	assert complexity>=2, f"complexity ({complexity}) should be >=2"
	if complexity==2:
		structures = [
					[-1,-1,'noun', 'intransverb', -1,-1,-1, -1,-1,-1, -1],
					]
	elif complexity==3:
		structures = [
					[-1,-1,'noun', 'transverb', -1,-1,'noun', -1,-1,-1,-1],
					['det',-1,'noun', 'intransverb', -1,-1,-1, -1,-1,-1, -1],
					[-1,'adj','noun', 'intransverb', -1,-1,-1, -1,-1,-1, -1],
					[-1,-1,'noun', 'intransverb', -1,-1,-1, -1,-1,-1, 'adv'],
					]
	elif complexity==4:
		structures = [
					['det',-1,'noun', 'transverb', -1,-1,'noun', -1,-1,-1,-1],
					[-1,'adj','noun', 'transverb', -1,-1,'noun', -1,-1,-1,-1],
					[-1,-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,-1],
					[-1,-1,'noun', 'transverb', -1,'adj','noun', -1,-1,-1,-1],
					[-1,-1,'noun', 'transverb', -1,-1,'noun', -1,-1,-1,'adv'],
					['det','adj','noun', 'intransverb', -1,-1,-1, -1,-1,-1, -1],
					['det',-1,'noun', 'intransverb', -1,-1,-1, -1,-1,-1, 'adv'],
					[-1,'adj','noun', 'intransverb', -1,-1,-1, -1,-1,-1, 'adv'],
					[-1,-1,'noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', -1],
					]
	elif complexity==5:
		structures = [
					['det','adj','noun', 'transverb', -1,-1,'noun', -1,-1,-1,-1],
					['det',-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,-1],
					['det',-1,'noun', 'transverb', -1,'adj','noun', -1,-1,-1,-1],
					['det',-1,'noun', 'transverb', -1,-1,'noun', -1,-1,-1,'adv'],
					[-1,'adj','noun', 'transverb', 'det',-1,'noun', -1,-1,-1,-1],
					[-1,'adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,-1],
					[-1,'adj','noun', 'transverb', -1,-1,'noun', -1,-1,-1,'adv'],
					[-1,-1,'noun', 'transverb', 'det','adj','noun', -1,-1,-1,-1],
					[-1,-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
					[-1,-1,'noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
					[-1,-1,'noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun',-1],
					['det','adj','noun', 'intransverb', -1,-1,-1, -1,-1,-1, 'adv'],
					['det',-1,'noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', -1],
					[-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', -1],
					[-1,-1,'noun', 'intransverb', -1,-1,-1, 'prep','det','noun', -1],
					[-1,-1,'noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],
					]
	elif complexity==6:
		structures = [
					['det','adj','noun', 'transverb', 'det',-1,'noun', -1,-1,-1,-1],
					['det','adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,-1],
					['det','adj','noun', 'transverb', -1,-1,'noun', -1,-1,-1,'adv'],
					['det',-1,'noun', 'transverb', 'det','adj','noun', -1,-1,-1,-1],
					['det',-1,'noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
					['det',-1,'noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
					[-1,'adj','noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
					[-1,'adj','noun', 'transverb', 'det','adj','noun', -1,-1,-1,-1],
					[-1,'adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
					[-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun',-1],
					[-1,-1,'noun', 'transverb', 'det','adj','noun', -1,-1,-1,'adv'],
					[-1,-1,'noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun',-1],
					[-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
					[-1,-1,'noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
					[-1,-1,'noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun','adv'],
					['det','adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', -1],
					['det',-1,'noun', 'intransverb', -1,-1,-1, 'prep','det','noun', -1],
					['det',-1,'noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],
					[-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep','det','noun', -1],
					[-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],
					[-1,-1,'noun', 'intransverb', -1,-1,-1, 'prep','det','noun', 'adv'],
					]
	elif complexity==7:
		structures = [
					['det','adj','noun', 'transverb', 'det','adj','noun', -1,-1,-1,-1],
					['det','adj','noun', 'transverb', 'det',-1,'noun', -1,-1,-1,'adv'],
					['det','adj','noun', 'transverb', -1,'adj','noun', -1,-1,-1,'adv'],
					['det','adj','noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun',-1],
					['det',-1,'noun', 'transverb', 'det','adj','noun', -1,-1,-1,'adv'],
					['det',-1,'noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun',-1],
					['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
					['det',-1,'noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
					['det',-1,'noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun','adv'],
					[-1,'adj','noun', 'transverb', 'det','adj','noun', -1,-1,-1,'adv'],
					[-1,'adj','noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun',-1],
					[-1,'adj','noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
					[-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
					[-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun','adv'],
					[-1,-1,'noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
					[-1,-1,'noun', 'transverb', 'det',-1,'noun', 'prep','det','noun',-1],
					[-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun','adv'],
					[-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun',-1],
					[-1,-1,'noun', 'transverb', -1,-1,'noun', 'prep','det','noun','adv'],
					['det','adj','noun', 'intransverb', -1,-1,-1, 'prep','det','noun', -1],
					['det','adj','noun', 'intransverb', -1,-1,-1, 'prep',-1,'noun', 'adv'],
					['det',-1,'noun', 'intransverb', -1,-1,-1, 'prep','det','noun', 'adv'],
					[-1,'adj','noun', 'intransverb', -1,-1,-1, 'prep','det','noun', 'adv'],
					]
	elif complexity==8:
		structures = [
					['det','adj','noun', 'transverb', 'det','adj','noun', -1,-1,-1,'adv'],
					['det','adj','noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun',-1],
					['det','adj','noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun',-1],
					['det','adj','noun', 'transverb', -1,-1,'noun', 'prep',-1,'noun','adv'],
					['det','adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun',-1],
					['det',-1,'noun', 'transverb', 'det','adj','noun', 'prep',-1,'noun',-1],
					['det',-1,'noun', 'transverb', 'det',-1,'noun', 'prep','det','noun',-1],
					['det',-1,'noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
					['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun',-1],
					['det',-1,'noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun','adv'],
					['det',-1,'noun', 'transverb', -1,-1,'noun', 'prep','det','noun','adv'],
					[-1,'adj','noun', 'transverb', 'det','adj','noun', 'prep',-1,'noun',-1],
					[-1,'adj','noun', 'transverb', 'det',-1,'noun', 'prep','det','noun',-1],
					[-1,'adj','noun', 'transverb', 'det',-1,'noun', 'prep',-1,'noun','adv'],
					[-1,'adj','noun', 'transverb', -1,'adj','noun', 'prep','det','noun',-1],
					[-1,'adj','noun', 'transverb', -1,'adj','noun', 'prep',-1,'noun','adv'],
					[-1,'adj','noun', 'transverb', -1,-1,'noun', 'prep','det','noun','adv'],
					[-1,-1,'noun', 'transverb', 'det','adj','noun', 'prep',-1,'noun','adv'],
					[-1,-1,'noun', 'transverb', 'det','adj','noun', 'prep','det','noun',-1],
					[-1,-1,'noun', 'transverb', 'det',-1,'noun', 'prep','det','noun','adv'],
					[-1,-1,'noun', 'transverb', -1,'adj','noun', 'prep','det','noun','adv'],
					['det','adj','noun', 'intransverb', -1,-1,-1, 'prep','det','noun', 'adv'],
					]
	if compositional and compositional_eval:
		assert complexity>3, f"there is no holdout structures to eval in compleixty 2,3, complexity ({complexity}) should be >3"
	keep_sampling = True
	while keep_sampling:
		struct = random.choice(structures) # choose a random sentence structure
		if not compositional: 
			keep_sampling = False
		elif compositional_eval: # comp and eval
			if struct in compositional_holdout:
				keep_sampling = False
		else: # comp and train
			if struct not in compositional_holdout:
				keep_sampling = False
	words, poss = [], [] # word ids, part of speech ids
	sentence = ""
	for r in struct:
		w = "_, "
		wid=-1
		rid=-1
		if r!=-1:
			w = random.choice(list(LEXEME_DICT[r]))
			wid = ALL_WORDS.index(w)
			rid = list(LEXEME_DICT.keys()).index(r)
			w += ", "
		sentence += w
		words.append(wid)
		poss.append(rid)
	assert len(poss)==len(words), f"len of poss {poss} should match words {words}"
	print(f"sample sentence with complexity {complexity}, struct {struct},\nsentence: {sentence}") if verbose else 0
	nwords = complexity # number of real words
	if len(poss)<max_sentence_length: # pad empty positions at the end, if any
		words += [-1]*(max_sentence_length-len(words))
		poss += [-1]*(max_sentence_length-len(poss))
	if spacing==False: # remove spacing within a sentence structure
		compactwords = []
		compactposs = []
		for i, (word, pos) in enumerate(zip(words, poss)):
			if word!=-1:
				assert pos!=-1, f"the {i}-th word and pos should both be nonempty\n{words}\n{poss}"
				compactwords.append(word)
				compactposs.append(pos)
		compactwords += [-1] * (max_sentence_length-len(compactwords)) # pad the end with empty 
		compactposs += [-1] * (max_sentence_length-len(compactposs))
		assert len(compactwords)==len(compactposs), f"length of compactwords {compactwords} and compactposs {compactposs} should equal"
		words = compactwords
		poss = compactposs
	return nwords, words, poss

def sample_episode(difficulty_mode, cur_curriculum_level, max_complexity, max_sentence_length, spacing, compositional, compositional_eval, compositional_holdout):
	'''
	Create a goal sentence for the episode
	Input
		difficulty_mode: {'max', 'curriculum', 'uniform' or -1, 1,2,3}
		cur_curriculum_level: {None, -1, 1,2,3}
	Return
		num_words: number of nonempty words in goal
		goal: [[lex ids], [lex types]], each of length max_sentence_length
	'''
	complexity = None # actual number of words in the stack, to be modified
	if difficulty_mode=='curriculum':
		assert cur_curriculum_level!=None, f"requested curriculum but current level is not given"
		if cur_curriculum_level==-1:
			complexity = random.randint(2, max_complexity)
		else:
			assert 1 <= cur_curriculum_level <= max_complexity, f"should have 1<= cur_curriculum_level ({cur_curriculum_level}) <= {mmax_sentence_lengthax_input_length}"
			population = list(range(2, max_complexity+1)) # possible number of words
			weights = np.zeros(len(population))
			weights[cur_curriculum_level-2] += 0.7 # weight for current level
			weights[max(cur_curriculum_level-3, 0)] += 0.15 # weight for the prev level
			weights[: max(cur_curriculum_level-3, 1)] += 0.15 / max(cur_curriculum_level-3, 1) # wieght for older levels
			assert np.sum(weights)==1, f"weights {weights} should sum to 1"
			complexity = random.choices(population=population, weights=weights, k=1)[0]
	elif difficulty_mode=='uniform' or (type(difficulty_mode)==int and difficulty_mode==-1): 
		complexity = random.randint(2, max_complexity)
	elif difficulty_mode=='max': 
		complexity = max_complexity
	elif type(difficulty_mode)==int:
		assert 1<=difficulty_mode<=max_complexity, \
			f"invalid difficulty mode: {difficulty_mode}, should be in set('max', 'uniform', -1, 'curriculum', 1,2,{max_complexity})"
		complexity = difficulty_mode
	else:
		raise ValueError(f"unrecognized difficulty mode {difficulty_mode} (type {type(difficulty_mode)})")
	assert complexity <= max_sentence_length, \
		f"number of actual words {complexity} should be smaller than max_sentence_length {max_sentence_length}"
	assert max_sentence_length==len(OUTPUT_FORMAT), f"max_sentence_length {max_sentence_length} should be {len(OUTPUT_FORMAT)}"
	num_words, goal, roles = sample_sentence(complexity=complexity, 
											max_sentence_length=max_sentence_length, 
											spacing=spacing, 
											compositional=compositional, 
											compositional_eval=compositional_eval, 
											compositional_holdout=compositional_holdout) 
	return num_words, goal, roles


def close_all_fibers_areas(simulator):
	actions = []
	# inhibit fibers
	for sidx, (a1,a2) in simulator.stateidx_to_fibername.items():
		if simulator.state[sidx]==1: # fiber is open
			actions.append(get_action_idx(('inhibit_fiber', a1, a2), simulator.action_dict))
	# inhibit areas
	for a in simulator.inhibitable_areas:
		sidx = simulator.area_to_stateidx[a]['opened']
		if simulator.state[sidx]==1 and a!=simulator.lexicon_area: # area is open
			actions.append(get_action_idx(('inhibit_area', a, None), simulator.action_dict))
	return actions

def expert_demo_language(simulator):
	final_actions = close_all_fibers_areas(simulator)
	curwid = simulator.last_active_assembly[LEX]
	action_dict = simulator.action_dict
	words, roles = simulator.goal, simulator.input_roles
	for wid, rid in zip(words, roles):
		if wid==-1:
			continue
		final_actions += go_activate(curwid, wid)
		r = list(LEXEME_DICT.keys())[rid]
		if r=='det':
			final_actions += parse_det(action_dict)
		elif r=='noun':
			final_actions += parse_noun(action_dict)
		elif r=='transverb':
			final_actions += parse_transverb(action_dict)
		# elif r=='prep':
		# 	final_actions += parse_prep(action_dict)
		elif r=='adj':
			final_actions += parse_adj(action_dict)
		elif r=='intransverb':
			final_actions += parse_intransverb(action_dict)
		# elif r=='adv':
		# 	final_actions += parse_adverb(action_dict)
		else:
			raise ValueError(f"role type {r} is not recognized")
		curwid = wid
	return final_actions

def synthetic_project(simulator, max_project_round=5, verbose=False):
	'''
	Strong project with symbolic assemblies.
	'''
	prev_last_active_assembly = copy.deepcopy(simulator.last_active_assembly) # {area: idx}
	prev_assembly_dict = copy.deepcopy(simulator.assembly_dict) # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
	new_num_assemblies = copy.deepcopy(simulator.num_assemblies) # current total number of assemblies in the brain
	state = copy.deepcopy(simulator.state)
	lexicon_area = simulator.lexicon_area
	prev_num_assemblies = None
	iround = 0 # current projection round
	all_visited = False # check if all opened areas are visited
	print(f"initial prev_assembly_dict\n{prev_assembly_dict}") if verbose else 0
	print(f'initial num_assemblies={new_num_assemblies}') if verbose else 0
	# Keep projecting while new assemblies are being created and other criteria hold. TODO: check endless loop condition
	while (new_num_assemblies != prev_num_assemblies) and (iround < max_project_round) and (not all_visited): 
		print(f"-------------------- new project round {iround}") if verbose else 0
		# Generate project map
		prev_num_assemblies = new_num_assemblies # update total number of assemblies
		receive_from = {} # {destination_area: [source_area1, source_area2, ...]}
		all_visited = False # whether all opened areas are visited 
		opened_areas = [] # open areas as dest in this round
		for idx in simulator.stateidx_to_fibername.keys(): # get opened fibers from state vector
			if state[idx]==1: # fiber is open
				area1, area2 = simulator.stateidx_to_fibername[idx] # get areas on both ends
				area1opened = state[simulator.area_to_stateidx[area1]['opened']]==1 
				area2opened = state[simulator.area_to_stateidx[area2]['opened']]==1
				if area1 != lexicon_area and area1opened and (area1 not in opened_areas): # skip if this is lex area
					opened_areas.append(area1)
				if area2 != lexicon_area and area2opened and (area2 not in opened_areas):
					opened_areas.append(area2)
				# check eligibility of areas, can only be source if there exists last active assembly in the area
				if (area1opened and area2opened) and (area1 != lexicon_area): #and (prev_last_active_assembly[area2] != -1): # lex area cannot receive
					receive_from[area1] = set([area2]).union(receive_from.get(area1, set())) # area2 source, area1 destination
				if (area1opened and area2opened) and (area2 != lexicon_area):# and (prev_last_active_assembly[area1] != -1): # bidirectional, area2 can also be source
					receive_from[area2] = set([area1]).union(receive_from.get(area2, set())) # area1 as source, to destination area2
		print(f'prev_assembly_dict: {prev_assembly_dict},\nprev_last_active_assembly: {prev_last_active_assembly},\nopened_areas: {opened_areas},\nreceive_from: {receive_from}') if verbose else 0
		# Do project
		assembly_dict = copy.deepcopy(prev_assembly_dict) # use assembly dict from prev round of project
		last_active_assembly = copy.deepcopy(prev_last_active_assembly) # use last activated assembly from prev round of project
		destinations = list(receive_from.keys())
		for area in simulator.all_areas: # process every destination area
			if area not in destinations:
				continue
			destination = area
			sources = []
			for s in list(receive_from[destination]):
				if prev_last_active_assembly[s]!=-1:
					sources.append(s)
			# sources = list(receive_from[destination]) # all input sources
			sources_permutation = list(itertools.permutations(sources)) # permutation of the sources, list of tuples
			active_assembly_id_in_destination = -1 # assume no matching connection exists by default
			print(f'{destination} as destination') if verbose else 0
			# search if destination area already has an assembly connected with input sources
			for assembly_idx, assembly_content in enumerate(prev_assembly_dict[destination]): # check existing assembly in dest one by one
				if prev_last_active_assembly[destination]==-1: # if dest is silence, skip
					continue
				connected_areas, connected_assembly_ids = assembly_content # assembly_content: [[Area1, Area2, ...], [assembly1, assembly2, ...]]
				print(f"\tchecking destination assembly id {assembly_idx}: connected areas {connected_areas}, connected ids {connected_assembly_ids}") if verbose else 0
				if (tuple(connected_areas) in sources_permutation): # if destination assembly connects with all source areas (only area names match)
					print("\t\tCandidate?") if verbose else 0
					match = True # now need to check if the assembly ids all match too
					for A, i in zip(connected_areas, connected_assembly_ids): # go through each area name and assembly id connected with this assembly
						if prev_last_active_assembly[A] != i: # assembly id does not match
							match = False
							print("\t\tBut ids do not match") if verbose else 0
					if match: # everything match, the exact connection from all input sources to destination already exists
						active_assembly_id_in_destination = assembly_idx 
						print("\t\tMatch!") if verbose else 0
						assert active_assembly_id_in_destination >= 0, f"\t\tFound matching connection between source and dest, but assembly_idx in dest is {active_assembly_id_in_destination}"
						break # exit the search
				if set(sources).issubset(set(connected_areas)): # if dest assembly connects with all source areas (only names match) and other areas
					print("\t\tCandidate (subset)?") if verbose else 0 # TODO: merge this if with previous if?
					match = True # now check if the assembly ids all match sources too 
					for A, i in zip(connected_areas, connected_assembly_ids): # go through each area name and assembly id connected with this assembly
						if (A in sources) and (prev_last_active_assembly[A] != i):
							match = False # assembly id does not match
					if match: # everything match, there is connection from all input sources (and some other areas) to destination
						active_assembly_id_in_destination = assembly_idx
						print("\t\tMatch!") if verbose else 0
						assert active_assembly_id_in_destination >= 0, f"\t\tFound matching connection between source and dest, but assembly_idx in dest is {active_assembly_id_in_destination}"
						break
			# no existing connection match, search if any of the sources is already connected with the latest activated assembly in dest
			if (active_assembly_id_in_destination == -1) and (prev_last_active_assembly[destination] != -1):
				dest_idx = prev_last_active_assembly[destination] # last activated assembly in dest
				connected_areas, connected_assembly_ids = prev_assembly_dict[destination][dest_idx][0], prev_assembly_dict[destination][dest_idx][1]
				print(f'\tsearching for partial candidates...\n\t\tlast activated assembly in dest: {dest_idx}, {prev_assembly_dict[destination][dest_idx]}') if verbose else 0
				for source in sources: # check if any source assemblies is connected with the last activated assembly in dest
					if (source in connected_areas) and (connected_assembly_ids[connected_areas.index(source)] == prev_last_active_assembly[source]): # both area name and index match
						print(f"\t\tPartial candidate Match! {source} {prev_last_active_assembly[source]}") if verbose else 0
						active_assembly_id_in_destination = dest_idx # all source assemblies will converge to the last activated assembly in dest
				if active_assembly_id_in_destination == -1: # partial candidate not found, search for non-optimal partial candidate (source assembly connects to a dest assembly that is not currently activated)
					print("\t\tnot found.\n\tsearching for non-optimal partial candidates...") if verbose else 0
					for source in sources: # check if any source assemblies connect to dest area
						if destination in prev_assembly_dict[source][prev_last_active_assembly[source]][0]:
							source_a_idx = prev_assembly_dict[source][prev_last_active_assembly[source]][0].index(destination) # locate the old dest assembly that connects with active source assembly
							active_assembly_id_in_destination = prev_assembly_dict[source][prev_last_active_assembly[source]][1][source_a_idx] # all source assemblies will converge to this old dest assembly
							print(f"\t\tnon-optimal partial candidate Match! source {source} {prev_last_active_assembly[source]} --> dest {active_assembly_id_in_destination}") if verbose else 0
			print(f"\tsearch ends, active_assembly_id_in_destination={active_assembly_id_in_destination}") if verbose else 0
			if active_assembly_id_in_destination == -1: # if still no existing connection match, create new assembly in destination
				assembly_dict[destination].append([sources, [prev_last_active_assembly[S] for S in sources]]) # [[A1, A2, ...], [a1, a2, ...]]
				active_assembly_id_in_destination = len(assembly_dict[destination])-1 # new assembly id
				new_num_assemblies += 1 # increment total number of assemblies in brain
				print(f'\tcreated new assembly in destination, id {active_assembly_id_in_destination}') if verbose else 0
			assert len(assembly_dict[destination]) > active_assembly_id_in_destination, f"new_dest_id={active_assembly_id_in_destination} out of bound of assembly_dict[destination]: {assembly_dict[destination]}"
			# reflect the newly activated destination assembly in source areas, update destination assembly dict if necessary
			for source in sources:
				if prev_last_active_assembly[source]==-1:
					continue
				print(f"\tchecking assembly dict for source {source}...") if verbose else 0
				match = False # checks if prev_assembly_dict[source][prev_last_active_assembly[source]] contains any assembly in dest
				for i, (A, a) in enumerate(zip(prev_assembly_dict[source][prev_last_active_assembly[source]][0], prev_assembly_dict[source][prev_last_active_assembly[source]][1])):
					print(f'\t\tlast active source is connected to: A={A}, a={a}') if verbose else 0
					if A==destination: # source already has a connection to dest, may need to update this connection
						match = True 
						if (a!=active_assembly_id_in_destination): # source connects with another assembly in dest
							updateidx = None # replace the connection, become source current active assembly --> dest current active assembly
							if destination in assembly_dict[source][prev_last_active_assembly[source]][0]: # using new dict as update may relate to the newly created dest assembly
								updateidx = assembly_dict[source][prev_last_active_assembly[source]][0].index(destination)
							if updateidx != None:
								assembly_dict[source][prev_last_active_assembly[source]][1][updateidx] = active_assembly_id_in_destination
								print("\t\tsource dict area match dest. Update source dict.") if verbose else 0
							# for symmetry, also check dest dict, remove the connection btw dest old assembly and source, add new connection if needed
							old_dest_id = prev_assembly_dict[source][prev_last_active_assembly[source]][1][i]
							new_dest_id = active_assembly_id_in_destination
							new_source_id = prev_last_active_assembly[source]
							for AA, aa in zip(prev_assembly_dict[destination][old_dest_id][0], prev_assembly_dict[destination][old_dest_id][1]):
								if (AA==source): # remove connection between the dest old assembly and the new source assembly
									popidx = None
									if AA in assembly_dict[destination][old_dest_id][0]:
										popidx = assembly_dict[destination][old_dest_id][0].index(AA)
										if assembly_dict[destination][old_dest_id][1][popidx]!= new_source_id:
											popidx = None
									if popidx != None:
										assembly_dict[destination][old_dest_id][0].pop(popidx)
										assembly_dict[destination][old_dest_id][1].pop(popidx)
										print(f"\t\tdest dict old id removed, old_dest_id={old_dest_id}, AA={AA}, aa={aa}") if verbose else 0
									# add new connection from dest assembly to activated source assembly
									if AA not in assembly_dict[destination][new_dest_id][0]: 
										assembly_dict[destination][new_dest_id][0].append(AA)
										assembly_dict[destination][new_dest_id][1].append(aa)
										print(f'\t\tdest dict new id added, new_dest_id={new_dest_id}, AA={AA}, aa={aa}') if verbose else 0
									else: # or update existing dest assembly
										idx = assembly_dict[destination][new_dest_id][0].index(AA)
										assembly_dict[destination][new_dest_id][1][idx] = aa
										print(f"\t\tdest dict new id updated, new_dest_id={new_dest_id}, AA={AA}, aa={aa}") if verbose else 0
				# check if new dest dict needs to be updated wrt source
				popidx = None
				for j, (AA, aa) in enumerate(zip(assembly_dict[destination][active_assembly_id_in_destination][0], assembly_dict[destination][active_assembly_id_in_destination][1])):
					print("\t\tchecking if dest dict needs to be updated wrt source") if verbose else 0
					if (AA==source) and (aa!=prev_last_active_assembly[source]): # if new dest is connected with a wrong id, remove wrong id
						old_source_id = assembly_dict[destination][active_assembly_id_in_destination][1][j]
						assembly_dict[destination][active_assembly_id_in_destination][1][j] = prev_last_active_assembly[source]
						print('\t\tdest dict updated, to new source a') if verbose else 0
						if destination in assembly_dict[source][old_source_id][0]:
							popidx = assembly_dict[source][old_source_id][0].index(destination)
							# if assembly_dict[source][old_source_id][1][popidx]!=prev_assembly_dict[source][prev_last_active_assembly[source]][1][i]:
							if assembly_dict[source][old_source_id][1][popidx]!=active_assembly_id_in_destination:
								print("\t\treset") if verbose else 0
								popidx = None
				if popidx != None:
					assembly_dict[source][old_source_id][0].pop(popidx)
					assembly_dict[source][old_source_id][1].pop(popidx)
					print(f"\t\tsource dict removed old_source_id {old_source_id}, popidx={popidx}, new source dict={assembly_dict[source]}") if verbose else 0
				# if prev_assembly_dict[source][prev_last_active_assembly[source]] is not connected with any assembly in dest 
				if not match: # append new dest to source dict 
					if destination not in assembly_dict[source][prev_last_active_assembly[source]][0]:
						assembly_dict[source][prev_last_active_assembly[source]][0].append(destination)
						assembly_dict[source][prev_last_active_assembly[source]][1].append(active_assembly_id_in_destination)
						print("\t\tsource dict added dest a") if verbose else 0
				# if dest is not connected with source at all, add the source to dest dict
				if source not in assembly_dict[destination][active_assembly_id_in_destination][0]:
					assembly_dict[destination][active_assembly_id_in_destination][0].append(source)
					assembly_dict[destination][active_assembly_id_in_destination][1].append(prev_last_active_assembly[source])
					print("\t\tdest dict did not have source a, added source a") if verbose else 0
				# check every assembly in the source
				visited = 0
				popa = []
				popidx = []
				for i, (Alist, alist) in enumerate(assembly_dict[source]):
					for j, (A, a) in enumerate(zip(Alist, alist)):
						if A==destination and a==active_assembly_id_in_destination:
							visited += 1
							if i != prev_last_active_assembly[source]:
								# if assembly i in source is connected with the dest, but dest is not connected with i, need to delete the connection
								visited -= 1
								popa.append(i)
								popidx.append(j)
				if len(popa)>0:
					for i, j in zip(popa[::-1], popidx[::-1]):
						print(f'\t\tsource dict popped assembly={i} connection={j}, in {assembly_dict[source][i]}') if verbose else 0
						assembly_dict[source][i][0].pop(j)
						assembly_dict[source][i][1].pop(j)
				# if no assembly connecting from source last active to destination active
				if visited==0 and (destination not in assembly_dict[source][prev_last_active_assembly[source]][0]): 
					# add the new connection
					assembly_dict[source][prev_last_active_assembly[source]][0].append(destination)
					assembly_dict[source][prev_last_active_assembly[source]][1].append(active_assembly_id_in_destination)
					print('\t\tsource dict added destination') if verbose else 0
				prev_assembly_dict = assembly_dict ##
				prev_last_active_assembly = last_active_assembly ##
			# update the last activated assembly in destination
			last_active_assembly[destination] = active_assembly_id_in_destination
			print(f'\ttotal number of assemblies={new_num_assemblies}') if verbose else 0
			# remove dest from opened area
			opened_areas.remove(destination)
			if len(opened_areas)==0:
				all_visited = True
				print('\tall_visited=True') if verbose else 0
			prev_assembly_dict = assembly_dict
			prev_last_active_assembly = last_active_assembly
		# Current project round completes, update assembly dict
		prev_assembly_dict = assembly_dict
		prev_last_active_assembly = last_active_assembly
		iround += 1
	# All project rounds complete
	num_assemblies = new_num_assemblies
	assembly_dict = copy.deepcopy(prev_assembly_dict)
	last_active_assembly = copy.deepcopy(prev_last_active_assembly)
	print(f"\n--------------------- end of all project rounds,\nnum_assemblies={num_assemblies},\nlast_active_assembly={last_active_assembly},\nassembly_dict:{assembly_dict}") if verbose else 0
	return assembly_dict, last_active_assembly, num_assemblies

def get_end_assembly_dict(simulator, actions):
	simulator.reset()
	for t, a in enumerate(actions):
		_, _, _, _, _ = simulator.step(a)
	return simulator.assembly_dict, simulator.last_active_assembly