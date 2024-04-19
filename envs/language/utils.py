import random
import pptree
import copy
import itertools
import pprint 

# BrainAreas
LEX = "LEX"
DET = "DET"
SUBJ = "SUBJ"
OBJ = "OBJ"
VERB = "VERB"
PREP = "PREP"
PREP_P = "PREP_P"
ADJ = "ADJ"
ADVERB = "ADVERB"
AREAS = [LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]
# AREAS = [LEX, DET, SUBJ, OBJ, VERB]
FIBERS = [(LEX, DET), 
			(LEX, SUBJ),
			(LEX, OBJ),
			(LEX, VERB),
			(LEX, ADJ),
			(LEX, ADVERB),
			(LEX, PREP),
			(LEX, PREP_P),

			(DET, SUBJ),
			(DET, OBJ),
			(DET, PREP_P),


			(ADJ, SUBJ),
			(ADJ, OBJ),
			(ADJ, PREP_P),

			(PREP_P, PREP),
			(PREP_P, VERB),
			(PREP_P, SUBJ),
			(PREP_P, OBJ),

			(VERB, SUBJ),
			(VERB, OBJ),
			(VERB, ADJ),
			(VERB, ADVERB),
			(VERB, PREP_P),

]
# all lexicon and their types
LEXEME_DICT = {
		'det': ['the', 'a'], 
		'noun': ['dogs', 'cats', 'mice', 'people', 'man', 'woman'], 
		'transverb': ['chase', 'love', 'bite', 'saw'], 
		'prep': ['of', 'in'], 
		'adj': ['big', 'bad'], 
		'intransverb': ['run', 'fly'], 
		'adv': ['quickly'], 
		'copula': ['are'],
		}
ALL_WORDS = [w for l in LEXEME_DICT.values() for w in l]
# readout methods
FIXED_MAP_READOUT = 1
FIBER_READOUT = 2
NATURAL_READOUT = 3
ENGLISH_READOUT_RULES = {
		VERB: [LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ],
		SUBJ: [LEX, DET, ADJ, PREP_P],
		OBJ: [LEX, DET, ADJ, PREP_P],
		PREP_P: [LEX, PREP, ADJ, DET],
		PREP: [LEX],
		ADJ: [LEX],
		DET: [LEX],
		ADVERB: [LEX],
		LEX: [],
	}


def init_simulator_areas():
	return AREAS, LEX, DET, FIBERS

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
	# actions to parse a generic noun
	action_tuples = [
				[
				("disinhibit_fiber", LEX, SUBJ),
				("disinhibit_fiber", LEX, OBJ),
				("disinhibit_fiber", LEX, PREP_P),
				("disinhibit_fiber", DET, SUBJ),
				("disinhibit_fiber", DET, OBJ),
				("disinhibit_fiber", DET, PREP_P),
				("disinhibit_fiber", ADJ, SUBJ),
				("disinhibit_fiber", ADJ, OBJ),
				("disinhibit_fiber", ADJ, PREP_P),
				("disinhibit_fiber", VERB, OBJ),
				("disinhibit_fiber", PREP_P, PREP),
				("disinhibit_fiber", PREP_P, SUBJ),
				("disinhibit_fiber", PREP_P, OBJ),
				],
				
				[("project_star", None, None),],

				[
				#("inhibit_area", DET, None),
				# ("inhibit_area", ADJ, None),
				# ("inhibit_area", PREP_P, None),
				# ("inhibit_area", PREP, None),
				("inhibit_fiber", LEX, SUBJ),
				("inhibit_fiber", LEX, OBJ),
				("inhibit_fiber", LEX, PREP_P),
				("inhibit_fiber", ADJ, SUBJ),
				("inhibit_fiber", ADJ, OBJ),
				("inhibit_fiber", ADJ, PREP_P),
				("inhibit_fiber", DET, SUBJ),
				("inhibit_fiber", DET, OBJ),
				("inhibit_fiber", DET, PREP_P),
				("inhibit_fiber", VERB, OBJ),
				("inhibit_fiber", PREP_P, PREP),
				("inhibit_fiber", PREP_P, VERB),
				("inhibit_fiber", PREP_P, SUBJ),
				("inhibit_fiber", PREP_P, OBJ),
				("inhibit_fiber", VERB, ADJ),
				("disinhibit_fiber", LEX, SUBJ),
				("disinhibit_fiber", LEX, OBJ),
				("disinhibit_fiber", DET, SUBJ),
				("disinhibit_fiber", DET, OBJ),
				("disinhibit_fiber", ADJ, SUBJ),
				("disinhibit_fiber", ADJ, OBJ),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_transverb(action_dict):
	# actions to parse transitive verb
	action_tuples = [
				[
				("disinhibit_fiber", LEX, VERB),
				("disinhibit_fiber", VERB, SUBJ),
				("disinhibit_fiber", VERB, ADVERB),
				# ("disinhibit_area", ADVERB, None),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, VERB),
				# ("disinhibit_area", OBJ, None),
				# ("inhibit_area", SUBJ, None),
				# ("inhibit_area", ADVERB, None),
				("disinhibit_fiber", PREP_P, VERB),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_intransverb(action_dict):
	action_tuples = [
				[
				("disinhibit_fiber", LEX, VERB),
				("disinhibit_fiber", VERB, SUBJ),
				("disinhibit_fiber", VERB, ADVERB),
				# ("disinhibit_area", ADVERB, None),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, VERB),
				# ("inhibit_area", SUBJ, None),
				# ("inhibit_area", ADVERB, None),
				("disinhibit_fiber", PREP_P, VERB),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_copula(action_dict):
	action_tuples = [
				[
				("disinhibit_fiber", LEX, VERB),
				("disinhibit_fiber", VERB, SUBJ),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, VERB),
				("disinhibit_fiber", ADJ, VERB),
				# ("disinhibit_area", OBJ, None),
				# ("inhibit_area", SUBJ, None),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_adverb(action_dict):
	action_tuples = [
				[
				# ("disinhibit_area", ADVERB, None),
				("disinhibit_fiber", LEX, ADVERB),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, ADVERB),
				# ("inhibit_area", ADVERB, None),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_det(action_dict):
	action_tuples = [
				[
				# ("disinhibit_area", DET, None),
				("disinhibit_fiber", LEX, DET),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, DET),
				("inhibit_fiber", VERB, ADJ),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_adj(action_dict):
	action_tuples = [
				[
				# ("disinhibit_area", ADJ, None),
				("disinhibit_fiber", LEX, ADJ),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, ADJ),
				("inhibit_fiber", VERB, ADJ),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def parse_prep(action_dict):
	action_tuples = [
				[
				("disinhibit_fiber", LEX, PREP),
				# ("disinhibit_area", PREP, None),
				],

				[("project_star", None, None),],

				[
				("inhibit_fiber", LEX, PREP),
				("inhibit_fiber", LEX, SUBJ),
				("inhibit_fiber", LEX, OBJ),
				("inhibit_fiber", DET, SUBJ),
				("inhibit_fiber", DET, OBJ),
				("inhibit_fiber", ADJ, SUBJ),
				("inhibit_fiber", ADJ, OBJ),
				# ("disinhibit_area", PREP, None),
				],
				]
	return get_action_idxs(action_tuples, action_dict)

def go_activate(curid, newid, action_goto_next=[17], action_goto_prev=[16]):
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

def check_prerequisite(arr, k, value=1):
	'''
	Check if the first k elements in arr are all equal to value.
	'''
	if len(arr) < k:
		raise ValueError(f"!!!Warning, in utils.check_prerequisite, k={k} is out of range {len(arr)}")
	for i in range(k):
		if arr[i] != value:
			return False
	return True

def all_fiber_closed(state, stateidx_to_fibername):
	'''
	Check whether all fibers in the state vector are closed.
	'''
	return np.all([state[i]==0 for i in stateidx_to_fibername.keys()])

def calculate_unit_reward(reward_decay_factor, num_items, episode_max_reward=1):
	return episode_max_reward / sum([reward_decay_factor**i for i in range(num_items)])

def calculate_readout_reward(readout, goal, correct_record, reward_decay_factor, empty_unit):
	assert len(readout) == len(goal), f"readout length {len(readout)} and goal length {len(goal)} should match."
	units = 0
	num_correct = 0
	prerequisite = [0 for _ in range(len(goal))] # the second block will received reward only if first block is also readout correctly
	for jblock in range(len(goal)): # read from top to bottom
		if readout[jblock] == goal[jblock]: # block match
			prerequisite[jblock] = 1
			num_correct += 1
			if correct_record[jblock]==0 and utils.check_prerequisite(prerequisite, jblock): # reward new correct blocks in this episode
				if readout[jblock]==-1: # empty position gets smaller reward
					units += empty_unit
				else: # actual block match gets larger reward (with decay)
					units += (reward_decay_factor**jblock) # reward scales by position
				correct_record[:jblock+1] = 1 # set the block record and all preceeding record to 1, record history for episode
	all_correct = (num_correct > 0) and (num_correct == len(goal))
	if all_correct:
		assert np.all([r==1 for r in correct_record])
	return units, all_correct, correct_record

def stimulate(source, destinations, assembly_dict, last_active_assembly):
	sourceaid = last_active_assembly[source]
	if sourceaid==-1: # source is silent
		return assembly_dict, last_active_assembly 
	for dest in destinations: # stimulate from source to each dest
		print(f"source {source}, dest {dest}, assembly_dict[source]: {assembly_dict[source]}, sourceaid {sourceaid}")
		if dest in assembly_dict[source][sourceaid][0]: # if source assembly is connected with dest
			destaid = assembly_dict[source][sourceaid][0].index(dest) # find the connected dest assembly
			last_active_assembly[dest] = destaid # update active assembly in dest
	return assembly_dict, last_active_assembly

def synthetic_readout(simulator, readout_method=FIXED_MAP_READOUT):
	assembly_dict = copy.deepcopy(simulator.assembly_dict)
	last_active_assembly = copy.deepcopy(simulator.last_active_assembly)
	dependencies = []
	def read(area, assembly_dict, last_active_assembly, mapping=ENGLISH_READOUT_RULES):
		destinations = mapping[area]
		assembly_dict, last_active_assembly = stimulate(source=area, destinations=destinations, assembly_dict=assembly_dict, last_active_assembly=last_active_assembly)
		thisword = ALL_WORDS[last_active_assembly[LEX]]
		for dest in destinations:
			if dest==LEX:
				continue
			assembly_dict, last_active_assembly = stimulate(source=area, destinations=[LEX], assembly_dict=assembly_dict, last_active_assembly=last_active_assembly)
			otherword = ALL_WORDS[last_active_assembly[LEX]]
			dependencies.append([thisword, (otherword, dest)])
		for dest in destinations:
			if dest != LEX:
				assembly_dict, last_active_assembly = read(dest, assembly_dict, last_active_assembly, mapping)
		return assembly_dict, last_active_assembly
	def treeify(parsed_dict, parent):
		for key, vals in parsed_dict.items():
			keynode = pptree.Node(key, parent)
			if isinstance(vals, str):
				_ = pptree.Node(vals, keynode)
			else:
				treeify(vals, keynode)
	if readout_method==FIXED_MAP_READOUT: # VERB --> SUBJ, OBJ, LEX
		assembly_dict, last_active_assembly = read(VERB, assembly_dict, last_active_assembly)
		print(f"dependencies: {dependencies}")
		parsed = {VERB: dependencies}
		root = pptree.Node(VERB)
		treeify(parsed[VERB], root)
	if readout_method==FIBER_READOUT:
		activated_fibers = get_open_fibers(simulator)
		assembly_dict, last_active_assembly = read(VERB, assembly_dict, last_active_assembly, mapping=activated_fibers)

	print(f"dependencies from readout {dependencies}")
	return dependencies

def decode_dependencies(dependencies):
	return ["word1", "word2"]

def get_open_fibers(simulator):
	return {source: [dest1, dest2]}

def sample_sentence(complexity, max_input_length):
	structures = [ [r] for r in LEXEME_DICT.keys()] # default single word
	if complexity==2:
		structures = [
					['noun', 'intransverb'],
					]
	elif complexity==3:
		structures = [
					['noun', 'transverb', 'noun'],
					['det', 'noun', 'intransverb'],
					['noun', 'intransverb', 'adv'],
					['adj', 'noun', 'intransverb'],
					['noun', 'copula', 'adj'],
					]
	struct = random.choice(structures)
	words, roles = [], []
	sentence = ""
	for r in struct:
		w = random.choice(list(LEXEME_DICT[r]))
		sentence += w
		wid = ALL_WORDS.index(w)
		words.append(wid)
		rid = list(LEXEME_DICT.keys()).index(r)
		roles.append(rid)
	assert len(roles)==len(words), f"len of roles {roles} should match words {words}"
	print(f"sample sentence with complexity {complexity}, struct {struct},\nsentence: {sentence}")
	nwords = len(words)
	if len(roles)<max_input_length: # pad empty positions
		words += [-1]*(max_input_length-len(words))
		roles += [-1]*(max_input_length-len(roles))
	return nwords, [words, roles]

def sample_episode(difficulty_mode, cur_curriculum_level, max_input_length, max_complexity=3):
	'''
	Create a goal sentence for the episode
	Input
		difficulty_mode: {'max', 'curriculum', 'uniform' or -1, 1,2,3}
		cur_curriculum_level: {None, -1, 1,2,3}
	Return
		num_words: number of nonempty words in goal
		goal: [[lex ids], [lex types]], each of length max_input_length
	'''
	complexity = None # actual number of blocks in the stack, to be modified
	if difficulty_mode=='curriculum':
		assert cur_curriculum_level!=None, f"requested curriculum but current level is not given"
		if cur_curriculum_level==-1:
			complexity = random.randint(1, max_complexity)
		else:
			assert 1 <= cur_curriculum_level <= max_complexity, f"should have 1<= cur_curriculum_level ({cur_curriculum_level}) <= {self.stack_max_blocks}"
			population = list(range(1, max_complexity+1)) # possible number of blocks
			weights = np.zeros(max_complexity)
			weights[cur_curriculum_level-1] += 0.7 # weight for current level
			weights[max(cur_curriculum_level-2, 0)] += 0.15 # weight for the prev level
			weights[: max(cur_curriculum_level-2, 1)] += 0.15 / max(cur_curriculum_level-2, 1)
			assert np.sum(weights)==1, f"weights {weights} should sum to 1"
			complexity = random.choices(population=population, weights=weights, k=1)[0]
	elif difficulty_mode=='uniform' or (type(difficulty_mode)==int and difficulty_mode==-1): 
		complexity = random.randint(1, max_complexity)
	elif difficulty_mode=='max': 
		complexity = max_complexity
	elif type(difficulty_mode)==int:
		assert 1<=difficulty_mode<=max_complexity, \
			f"invalid difficulty mode: {difficulty_mode}, should be in set('max', 'uniform', -1, 'curriculum', 1,2,{max_complexity})"
		complexity = difficulty_mode
	else:
		raise ValueError(f"unrecognized difficulty mode {difficulty_mode} (type {type(difficulty_mode)})")
	assert complexity <= max_input_length, \
		f"number of actual words {complexity} should be smaller than max_input_length {max_input_length}"
	num_words, goal = sample_sentence(complexity, max_input_length) 
	return num_words, goal

def expert_demo_language(simulator):
	def close_all_fibers():
		actions = []
		for sidx, (a1,a2) in simulator.stateidx_to_fibername.items():
			if simulator.state[sidx]==1: # fiber is open
				actions.append(get_action_idx(('inhibit_fiber', a1, a2), simulator.action_dict))
		return actions
	final_actions = close_all_fibers()
	curwid = simulator.last_active_assembly[LEX]
	action_dict = simulator.action_dict
	[words, roles] = simulator.goal
	for wid, rid in zip(words, roles):
		final_actions += go_activate(curwid, wid)
		r = list(LEXEME_DICT.keys())[rid]
		if r=='det':
			final_actions += parse_det(action_dict)
		elif r=='noun':
			final_actions += parse_noun(action_dict)
		elif r=='transverb':
			final_actions += parse_transverb(action_dict)
		elif r=='prep':
			final_actions += parse_prep(action_dict)
		elif r=='adj':
			final_actions += parse_adj(action_dict)
		elif r=='intransverb':
			final_actions += parse_intransverb(action_dict)
		elif r=='adv':
			final_actions += parse_adverb(action_dict)
		elif r=='copula':
			final_actions += parse_copula(action_dict)
		else:
			raise ValueError(f"role type {r} is not recognized")
		curwid = wid
	return final_actions

def synthetic_project(simulator, max_project_round=1, verbose=True):
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
		opened_areas = {} # open areas as dest in this round
		for idx in simulator.stateidx_to_fibername.keys(): # get opened fibers from state vector
			if state[idx]==1: # fiber is open
				area1, area2 = simulator.stateidx_to_fibername[idx] # get areas on both ends
				if area1 != lexicon_area: # skip if this is lex area
					opened_areas = set([area1]).union(opened_areas)
				if area2 != lexicon_area:
					opened_areas = set([area2]).union(opened_areas)
				# check eligibility of areas, can only be source if there exists last active assembly in the area
				if (prev_last_active_assembly[area1] != -1) and (area2 != lexicon_area): # lex area cannot receive
					receive_from[area2] = set([area1]).union(receive_from.get(area2, set())) # area1 as source, to destination area2
				if (prev_last_active_assembly[area2] != -1) and (area1 != lexicon_area): # bidirectional, area2 can also be source
					receive_from[area1] = set([area2]).union(receive_from.get(area1, set())) # area2 source, area1 destination
		print(f'prev_assembly_dict: {prev_assembly_dict},\nprev_last_active_assembly: {prev_last_active_assembly},\nopened_areas: {opened_areas},\nreceive_from: {receive_from}') if verbose else 0
		# Do project
		assembly_dict = copy.deepcopy(prev_assembly_dict) # use assembly dict from prev round of project
		last_active_assembly = copy.deepcopy(prev_last_active_assembly) # use last activated assembly from prev round of project
		destinations = list(receive_from.keys())
		destinations.sort(reverse=True) # head will be the last
		for destination in destinations: # process every destination area
			sources = list(receive_from[destination]) # all input sources
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
			# update the last activated assembly in destination
			last_active_assembly[destination] = active_assembly_id_in_destination
			print(f'\ttotal number of assemblies={new_num_assemblies}') if verbose else 0
			# remove dest from opened area
			opened_areas.remove(destination)
			if len(opened_areas)==0:
				all_visited = True
				print('\tall_visited=True') if verbose else 0
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

