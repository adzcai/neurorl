import copy
import itertools
import pprint
import random
import numpy as np

from envs.blocksworld.AC import bw_apps
from envs.blocksworld import test_puzzles

'''
DONE	
	holdout test set
	implement curriculum for parse
	implement curriculum for plan
'''


def init_simulator_areas(max_stacks=1, 
							max_node_areas=3, 
							prefix="G", 
							relocated_area="RELOCATED", 
							blocks_area="BLOCKS"):
	'''
	Initialize the list of area names in the brain, and the head area name.
	Intended for parse, remove, add Simulator.
	Input:
		max_stacks: (int = 1)
			max number of stacks that a brain can represent. 
			should be 1 for parse/add/remove/plan simulator.
		max_node_areas: (int = 3)
			number of node areas in the brain, not including the head area.
		prefix: (string = "G")
			prefix attached to each area name
		relocated_area: (string = "RELOCATED")
			name of the relocated area in the brain.
			usually this area is ignored in the simulator.
		blocks_area: (string = "BLOCKS")
			name of the blocks area (i.e. the lexicon area)
	Return:
		all_areas: (list of strings)
			list containing the names of all areas in the brain (including head, nodes, blocks, and relocated)
		head: (string)
			the name of the head area
		prefixed_node_areas
		relocated_area
		blocks_area
	'''
	assert max_stacks == 1, f"envs.blocksworld.utils max_stacks should be 1, but has {max_stacks}"
	# generate names for node areas
	node_areas = []
	for j in range(max_stacks):
		node_areas_stack_j = []
		for k in range(max_node_areas):
			node_areas_stack_j.append(str(j)+"_N"+str(k))
		node_areas.append(node_areas_stack_j)
	node_areas = node_areas
	head_areas = []
	for j in range(max_stacks):
		head_areas.append(str(j)+"_H")
	regions = [] # all areas together
	for j in range(max_stacks):
		regions_stack_j = node_areas[j] + [head_areas[j]]
		regions.append(regions_stack_j)
	other_areas = bw_apps.add_prefix(regions=[item for sublist in regions for item in sublist], prefix=prefix)
	other_areas = other_areas + [relocated_area]
	all_areas = [blocks_area]
	all_areas.extend(other_areas) 
	head = [element for element in all_areas if '_H' in element][0]
	prefixed_node_areas = bw_apps.add_prefix(regions=[item for sublist in node_areas for item in sublist], prefix=prefix)
	return all_areas, head, prefixed_node_areas, relocated_area, blocks_area


def go_activate_block(curblock, newblock, action_goto_next=[19], action_goto_prev=[20]):
	'''
	Input:
		curblock: (int)
			currently activated block assembly id in BLOCKS area
			should be in range [-1, max blocks in brain]
		newblock: (int)
			the new blockid to be activated
		action_goto_next: (list of int)
			a list of action indices to activate the next block assembly
		action_goto_prev: (list of int)
			a list of action indices to activate the previous block assembly
	Return:
		actions: (list of int)
			list of action indices to activate newblock id
	'''
	actions = []
	diff = curblock-newblock
	while diff != 0:
		if diff > 0:
			actions += action_goto_prev
			curblock -= 1
		else:	
			actions += action_goto_next
			curblock += 1
		diff = curblock-newblock
	return actions


def top(assembly_dict, last_active_assembly, head, blocks_area="BLOCKS"):
	'''
	Get the top area in the brain, this will be the area that directly connects from head area.
	Input: (can be retrieved from Simulator)
		assembly_dict: (dict) 
			dictionary of assembly associations that currently exist in the brain
			{area: [assembly_idx0[source areas[A0, A1], source assembly_idx[a0, a1]], 
					assembly_idx1[[A3], [a3]], 
					assembly_idx2[[A4], [a4]], 
					...]}
			i.e. area has assembly_idx0, which is associated/projected from area A0 assembly a0, and area A1 assembly a1
		last_active_assembly: (dict) 
			the latest activated assembly idx in each area
			{area: assembly_idx}
			assembly_idx = -1 means that no previously activated assembly exists
		head: (string) 
			the head node area's name in the brain
	Return: 
		area: (string)
			top area name
		a: (int)
			top area assembly id
		bid: (int)
			top block idx
	'''
	if last_active_assembly[head] == -1: # head is silence
		return None, None, None
	candidate_areas, candidate_as = assembly_dict[head][last_active_assembly[head]]
	for area, a in zip(candidate_areas, candidate_as):
		if blocks_area in assembly_dict[area][a][0]:
			idx = assembly_dict[area][a][0].index(blocks_area)
			bid = assembly_dict[area][a][1][idx]
			return area, a, bid
	return None, None, None


def is_last_block(assembly_dict, head, top_area, top_area_a, blocks_area="BLOCKS"):
	'''
	Check if top_area_a assembly in top_area represents the last block in the chain
	'''
	if top_area==None: # no top area given
		return False
	# check if top assembly connects with another node assembly that represents any block
	for A, a in zip(assembly_dict[top_area][top_area_a][0], assembly_dict[top_area][top_area_a][1]): 
		if (A != blocks_area and A != head) \
			and (('_N0' in top_area and '_N1' in A) or ('_N1' in top_area and '_N2' in A) or ('_N2' in top_area and '_N0' in A)) \
			and (blocks_area in assembly_dict[A][a][0]): 
				return False # top_area assembly connects with another node assembly that connects with blocks
	return True


def all_fiber_closed(state, stateidx_to_fibername):
	'''
	Check whether all fibers in the state vector are closed.
	Input:
		state: (list or numpy array of float32)
		stateidx_to_fibername: (dict)
			mapping state vector index to fiber between two areas
			{state idx: (area1, area2)}
	Return: True or False
	'''
	return np.all([state[i]==0 for i in stateidx_to_fibername.keys()])


def synthetic_readout(assembly_dict, 
						last_active_assembly, 
						head, 
						readout_length, 
						blocks_area="BLOCKS"):
	'''
	Read out the current chain of blocks representation from the brain. (Assuming the brain only represents max 1 stack.)
	Input: (most of them can be retrieved from Simulator)
		assembly_dict: (dict) 
			assembly associations that currently exist in the brain
			{area: [assembly_idx0[source areas[A0, A1], source assembly_idx[a0, a1]], 
					assembly_idx1[[A3], [a3]], 
					assembly_idx2[[A4], [a4]], 
					...]}
			i.e. area has assembly_idx0, which is associated/projected from area A0 assembly a0, and area A1 assembly a1
		last_active_assembly: (dict) 
			the latest activated assembly idx in each area
			{area: assembly_idx}
			assembly_idx = -1 means that no previously activated assembly exists
		head: (string) 
			the head node area's name in the brain
		readout_length: (int) 
			length of the readout list (i.e. number of blocks)
	Return:
		readout: (list of int) 
			list of blocks readout from the brain (chained from head/top to bottom).
			will be of length readout_length. invalid/empty block locations will be filled with None.
	'''
	readout = [] # list of blocks (assuming only 1 stack in the brain)
	if len(assembly_dict[head])==0 or last_active_assembly[head]==-1:
		return [None] * readout_length # no assembly in head, return [None, None, ...]
	# if assembly exists in head, get the first node connected with head
	areas_from_head, aidx_from_head = assembly_dict[head][last_active_assembly[head]][0], assembly_dict[head][last_active_assembly[head]][1]
	prev_area, prev_area_a = head, last_active_assembly[head]
	area, area_a = None, None # initiate next area to decode from
	if len(areas_from_head) != 0 and len(aidx_from_head)!= 0: # if head assembly is connected with a node area
		area, area_a = areas_from_head[0], aidx_from_head[0] # next area to read from
	for iblock in range(readout_length): 
		if area==None and area_a==None: # if current area is not available
			readout.append(None)
			continue
		elif blocks_area in assembly_dict[area][area_a][0]: # if current area is connected with blocks_area, decode
			ba = assembly_dict[area][area_a][0].index(blocks_area)
			bidx = assembly_dict[area][area_a][1][ba]
			readout.append(bidx)
		else: # current area is not connected with blocks_area
			readout.append(None)
		# find the next area to decode
		areas_from_area, aidx_from_area = assembly_dict[area][area_a] # assemblies connected with current area
		new_area, new_area_a = None, None
		for A, a in zip(areas_from_area, aidx_from_area): # iterate through current assembly's connections
			if (A != blocks_area) and (A != prev_area) and (A != head) \
				and (('_N0' in area and '_N1' in A) or ('_N1' in area and '_N2' in A) or ('_N2' in area and '_N0' in A)): 
				new_area = A # only look for the next node area in the correct order
				new_area_a = a
				break
		prev_area, prev_area_a = area, area_a
		area, area_a = new_area, new_area_a
	return readout


def check_prerequisite(arr, k, value=1):
	'''
	Check if the first k elements in arr are all equal to value.
	Input: 
		arr: (list or numpy array)
		k: (int)
		value: (int or float)
	'''
	if len(arr) < k:
		raise ValueError(f"!!!Warning, in utils.check_prerequisite, k={k} is out of range {len(arr)}")
	for i in range(k):
		if arr[i] != value:
			return False
	return True


def calculate_unit_reward(reward_decay_factor, 
							num_items, 
							episode_max_reward=1):
	'''
	Calculate the unit reward to use for the episode of 1 stack.
	This will ensure that the total episode reward will be episode_max_reward + action costs.
	Assume there will be reward given for each correct item in the list (e.g. each block correct in parse/add/remove/plan).
	Input:
		reward_decay_factor: (float > 0)
			reward for getting item at index i correct = unit_reward * (reward_decay_factor**i).
			for descending reward (first index is most rewarding), use 0 < factor < 1
			for ascending reward (last index is most reward), use factor > 1
		num_items: (int)
			total number of items that can receive correct reward in the episode.
			e.g. stack_max_blocks, puzzle_max_blocks
		episode_max_reward: (float = 1)
	Return:
		unit_reward: (float)
	'''
	return episode_max_reward / sum([reward_decay_factor**i for i in range(num_items)])


def calculate_readout_reward(readout, 
								goal, 
								correct_record, 
								reward_decay_factor):
	'''
	Calculate score by comparing current stack readout with goal stack.
		Reward decays from top to bottom block.
		Reward a block only if all its previous (higher) blocks are correct.
	Input: 
		readout: (list or numpy array of int)
			current readout from the brain (chain of blocks from top to bottom)
		goal: (list of int)
			the goal chain of blocks (from top/high to bottom/low) to be matched
		correct_record: (numpy array with binary values)
			binary record fro how many blocks are already correct in this episode (index 0 records the correctness of block idx 0).
			will not get reward anymore if the index was already correct in the episode.
		reward_decay_factor: (float)
			scaling factor applied on the unit_reward when assigning reward.
			each correct index will get a reward that is exponential to the unit_reward by this factor.
			for example, reward for getting index i correct = unit_reward * (reward_decay_factor**i).
			for descending reward (first index is most rewarding), use 0 < factor < 1
			for ascending reward (last index is most reward), use factor > 1
	Return: 
		score: (float)
			units of reward to award. 
			an unit is the smallest amount of reward to give for a correct index.
		all_correct: (boolean)
			whether current readout has all blocks correct
		correct_record: (numpy array with binary values)
			history of correct in the episode.
			note that correct_record all values==1 does not equal all_correct=True for the current readout.
			(e.g. got all blocks correct in the past but mess up something later)
	'''
	assert len(readout) == len(goal), f"readout length {len(readout)} and goal length {len(goal)} should match."
	score = 0
	num_correct = 0
	prerequisite = [0 for _ in range(len(goal))] # the second block will received reward only if first block is also readout correctly
	for jblock in range(len(goal)): # read from top to bottom
		if readout[jblock] == goal[jblock]: # block match
			prerequisite[jblock] = 1
			num_correct += 1
			if correct_record[jblock]==0 and check_prerequisite(prerequisite, jblock): # reward new correct blocks in this episode
				score += (reward_decay_factor**jblock) # reward scales by position
				correct_record[:jblock+1] = 1 # set the block record and all preceeding record to 1, record history for episode
	all_correct = (num_correct > 0) and (num_correct == len(goal))
	if all_correct:
		assert np.all([r==1 for r in correct_record])
	return score, all_correct, correct_record


def synthetic_project(state, 
						assembly_dict, 
						stateidx_to_fibername, 
						last_active_assembly,
						num_assemblies, 
						verbose=False, 
						max_project_round=5, 
						blocks_area="BLOCKS"):
	'''
	Strong project with symbolic assemblies.
	Input: (most of them can be retrieved from Simulator)
		state: (numpy array with float32)
			state vector, used for forming projection map
		assembly_dict: (dict)
			dictionary storing assembly associations that currently exist in the brain
			{area: [assembly_idx0[source areas[A0, A1], source assembly_idx[a0, a1]], 
					assembly_idx1[[A3], [a3]], 
					assembly_idx2[[A4], [a4]], 
					...]}
			i.e. area has assembly_idx0, which is associated/projected from area A0 assembly a0, and area A1 assembly a1
		stateidx_to_fibername: (dict)
			mapping state vector index to fiber between two areas
			{state idx: (area1, area2)}
		last_active_assembly: (dict) 
			dictionary storing the latest activated assembly idx in each area
			{area: assembly_idx}
			assembly_idx = -1 means that no previously activated assembly exists
		num_assemblies: (int)
			total number of assemblies exist in the brain (including blocks lexicon)
		verbose: (boolean=False) 
			True or False
		max_project_round: (int=5)
			maximum number of project rounds (usually <=5 is enough).
	Return:
		assembly_dict: (dict)
		last_active_assembly: (dict)
		num_assemblies: (int)
	'''
	prev_last_active_assembly = copy.deepcopy(last_active_assembly) # {area: idx}
	prev_assembly_dict = copy.deepcopy(assembly_dict) # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
	new_num_assemblies = num_assemblies # current total number of assemblies in the brain
	prev_num_assemblies = None
	iround = 0 # current projection round
	all_visited = False # check if all opened areas are visited
	print(f'initial num_assemblies={new_num_assemblies}') if verbose else 0
	# Keep projecting while new assemblies are being created and other criteria hold. TODO: check endless loop condition
	while (new_num_assemblies != prev_num_assemblies) and (iround <= max_project_round) and (not all_visited): 
		print(f"-------------------- new project round {iround}") if verbose else 0
		# Generate project map
		prev_num_assemblies = new_num_assemblies # update total number of assemblies
		receive_from = {} # {destination_area: [source_area1, source_area2, ...]}
		all_visited = False # whether all opened areas are visited 
		opened_areas = {} # open areas in this round
		for idx in stateidx_to_fibername.keys(): # get opened fibers from state vector
			if state[idx]==1: # fiber is open
				area1, area2 = stateidx_to_fibername[idx] # get areas on both ends
				if area1 != blocks_area: # skip if this is blocks area
					opened_areas = set([area1]).union(opened_areas)
				if area2 != blocks_area:
					opened_areas = set([area2]).union(opened_areas)
				# check eligibility of areas, can only be source if there exists last active assembly in the area
				if (prev_last_active_assembly[area1] != -1) and (area2 != blocks_area): # blocks area cannot receive
					receive_from[area2] = set([area1]).union(receive_from.get(area2, set())) # area1 as source, to destination area2
				if (prev_last_active_assembly[area2] != -1) and (area1 != blocks_area): # bidirectional, area2 can also be source
					receive_from[area1] = set([area2]).union(receive_from.get(area1, set())) # area2 source, area1 destination
		print(f'prev_last_active_assembly: {prev_last_active_assembly}, opened_areas: {opened_areas}, receive_from: {pprint.pprint(receive_from)}, prev_assembly_dict: {pprint.pprint(prev_assembly_dict)}') if verbose else 0
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
	print(f"\n--------------------- end of all project rounds, num_assemblies={num_assemblies}, last_active_assembly={last_active_assembly}, assembly_dict:{pprint.pprint(assembly_dict)}") if verbose else 0
	return assembly_dict, last_active_assembly, num_assemblies


def expert_demo_parse(goal, num_blocks):
	'''
	Generate random expert demonstration for parsing a stack of blocks.
	Input:
		goal: (list of length stack_max_blocks)
			consists of nonnegative ints (block ids) and None (empty block position)
			list as top to bottom block in the stack
		num_blocks: (int)
			valid number of blocks in the parse goal
	Return:
		expert_demo: (list of int)
			list of expert actions to parse the blocks
	'''
	stack = goal[:num_blocks]
	actions = []
	action_module0 = [10,0] # optimal fiber actions for parsing first block
	action_module1 = [11,1,6,2] # ... second block
	action_module2 = [7,3,12,4] # ... third block
	action_module3 = [13,5,0,8] # ... fourth block
	action_module4 = [1,9,2,6] # ... fifth block
	inhibit_action_module0 = [11, 1] # close fibers to terminate episode after module0
	inhibit_action_module1 = [7, 3] 
	inhibit_action_module2 = [13, 5]
	inhibit_action_module3 = [1, 9]
	inhibit_action_module4 = [3, 7]
	project_star = [18]
	activate_next_block = [19]
	activate_prev_block = [20]
	activated_block = -1 # currently activated block id in BLOCKS area
	if len(stack)>0: # module 0
		tmp_actions = []
		tmp_actions += go_activate_block(activated_block, stack[0], activate_next_block, activate_prev_block)
		tmp_actions += action_module0
		random.shuffle(tmp_actions)
		actions += tmp_actions
		actions += project_star
		activated_block = stack.pop(0)
	if len(stack)>0: # module 1
		tmp_actions = []
		tmp_actions += go_activate_block(activated_block, stack[0], activate_next_block, activate_prev_block)
		tmp_actions += action_module1
		random.shuffle(tmp_actions)
		actions += tmp_actions
		actions += project_star
		activated_block = stack.pop(0)
	else: # no more blocks after module0
		tmp_actions = inhibit_action_module0
		random.shuffle(tmp_actions)
		actions += tmp_actions
		return actions
	if len(stack)==0: # no more blocks after module1
		tmp_actions = inhibit_action_module1
		random.shuffle(tmp_actions)
		actions += tmp_actions
		return actions
	imodule = 0
	while True: # loop through module 2,3,4
		tmp_actions = []
		tmp_actions += go_activate_block(activated_block, stack[0], activate_next_block, activate_prev_block)
		if imodule%3 == 0:
			actions += action_module2
		elif imodule%3 == 1:
			actions += action_module3
		elif imodule%3 == 2:
			actions += action_module4
		random.shuffle(tmp_actions)
		actions += tmp_actions
		actions += project_star
		activated_block = stack.pop(0)
		if len(stack)==0:
			tmp_actions = [inhibit_action_module2, inhibit_action_module3, inhibit_action_module4][imodule%3]
			random.shuffle(tmp_actions)
			actions += tmp_actions
			return actions
		imodule += 1
	

def expert_demo_remove(simulator):
	'''
	Generate random expert demonstration for removing a block from a stack.
	Input:
		simulator: (remove.Simulator)
			simulator.goal (length stack_max_blocks, ints and None) is goal stack after removing the top
	Return:
		expert_demo: (list of int)
			list of expert actions to remove a block
	'''
	final_actions = []
	top_area_name, top_area_a, top_block_idx = top(simulator.assembly_dict, simulator.last_active_assembly, simulator.head, simulator.blocks_area)
	# first check if any fiber needs to be inhibited
	final_actions = []
	for stateidx in simulator.stateidx_to_fibername:
		if simulator.state[stateidx] == 1: # fiber opened
			aname, arg1, arg2 = "inhibit_fiber", simulator.stateidx_to_fibername[stateidx][0], simulator.stateidx_to_fibername[stateidx][1]
			try:
				final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
			except:
				final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(final_actions)
	# check is last block
	if simulator.goal[0]==None or (is_last_block(simulator.assembly_dict, simulator.head, top_area_name, top_area_a, simulator.blocks_area)): 
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(("silence_head", None))] )
		random.shuffle(final_actions)
		return final_actions
	# if not last block, get the new top area
	if '_N0' in top_area_name:
		new_top_area = [area for area in simulator.all_areas if '_N1' in area][0]
	elif '_N1' in top_area_name:
		new_top_area = [area for area in simulator.all_areas if '_N2' in area ][0]
	elif '_N2' in top_area_name:
		new_top_area = [area for area in simulator.all_areas if '_N0' in area ][0]
	# stimulate existing assemblies
	aname, arg1, arg2 = "disinhibit_fiber", simulator.head, top_area_name
	try:
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	tmp_actions = []
	aname, arg1, arg2 = "inhibit_fiber", simulator.head, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "disinhibit_fiber", simulator.blocks_area, new_top_area
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	tmp_actions += go_activate_block(simulator.last_active_assembly[simulator.blocks_area], simulator.goal[0])
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	aname, arg1, arg2 = "disinhibit_fiber", top_area_name, new_top_area 
	try:
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	tmp_actions = []
	aname, arg1, arg2 = "inhibit_fiber", top_area_name, new_top_area
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	# build new connection
	aname, arg1, arg2 = "disinhibit_fiber", new_top_area, simulator.head
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	action = ("silence_head", None)
	tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	# close all connections
	tmp_actions = []
	aname, arg1, arg2 = "inhibit_fiber", new_top_area, simulator.head
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "inhibit_fiber", simulator.blocks_area, new_top_area
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	return final_actions


def expert_demo_add(simulator):
	'''
	Generate random expert demonstration for adding a block to a stack
	Input:
		simulator: (add.Simulator)
			will contain attribute simulator.newblock that represents new block id
			and simulator.goal (length stack_max_blocks, ints and None) of goal stack after adding
	Return:
		expert_demo: (list of int)
			list of expert actions to add a block
	'''
	final_actions = []
	top_area_name, top_area_a, top_block_idx = top(simulator.assembly_dict, simulator.last_active_assembly, simulator.head, simulator.blocks_area)
	# first check if any fiber needs to be inhibited
	for stateidx in simulator.stateidx_to_fibername:
		if simulator.state[stateidx] == 1: # fiber opened
			aname, arg1, arg2 = "inhibit_fiber", simulator.stateidx_to_fibername[stateidx][0], simulator.stateidx_to_fibername[stateidx][1]
			try:
				final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
			except:
				final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	# activate new block assembly
	final_actions += go_activate_block(curblock=simulator.last_active_assembly[simulator.blocks_area], newblock=simulator.newblock)
	# get the new top area
	if top_area_name==None: # if no block exists in brain
		new_top_area = [area for area in simulator.all_areas if '_N0' in area][0] 
		aname, arg1, arg2 = "disinhibit_fiber", simulator.blocks_area, new_top_area
		try:
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
		except:
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
		random.shuffle(final_actions)
		action = ("project_star", None)
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
		tmp_actions = []
		if simulator.last_active_assembly[simulator.head] != -1:
			action = ("silence_head", None)
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
		aname, arg1, arg2 = "inhibit_fiber", simulator.blocks_area, new_top_area
		try:
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
		except:
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
		aname, arg1, arg2 = "disinhibit_fiber", new_top_area, simulator.head
		try:
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
		except:
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
		random.shuffle(tmp_actions)
		final_actions += tmp_actions
		action = ("project_star", None)
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
		tmp_actions = []
		aname, arg1, arg2 = "inhibit_fiber", new_top_area, simulator.head
		try:
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
		except:
			tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
		random.shuffle(tmp_actions)
		final_actions += tmp_actions
		return final_actions
	elif '_N0' in top_area_name:
		new_top_area = [area for area in simulator.all_areas if '_N2' in area][0]
	elif '_N1' in top_area_name:
		new_top_area = [area for area in simulator.all_areas if '_N0' in area ][0]
	elif '_N2' in top_area_name:
		new_top_area = [area for area in simulator.all_areas if '_N1' in area ][0]
	# encode new block in new top area
	aname, arg1, arg2 = "disinhibit_fiber", simulator.blocks_area, new_top_area
	try:
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(final_actions)
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	# stimulate head
	tmp_actions = []
	tmp_actions += go_activate_block(curblock=simulator.newblock, newblock=simulator.goal[1])
	aname, arg1, arg2 = "inhibit_fiber", simulator.blocks_area, new_top_area
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "disinhibit_fiber", simulator.blocks_area, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	tmp_actions = []
	aname, arg1, arg2 = "disinhibit_fiber", simulator.head, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "inhibit_fiber", simulator.blocks_area, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	tmp_actions = []
	aname, arg1, arg2 = "disinhibit_fiber", simulator.blocks_area, new_top_area
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "disinhibit_fiber", new_top_area, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	tmp_actions += go_activate_block(curblock=simulator.goal[1], newblock=simulator.newblock)
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	tmp_actions = []
	# project new block to head
	aname, arg1, arg2 = "inhibit_fiber", simulator.head, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "disinhibit_fiber", new_top_area, simulator.head
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	action = ("silence_head", None)
	tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	aname, arg1, arg2 = "inhibit_fiber", new_top_area, top_area_name
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	action = ("project_star", None)
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	# close all fibers
	tmp_actions = []
	aname, arg1, arg2 = "inhibit_fiber", new_top_area, simulator.head
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	aname, arg1, arg2 = "inhibit_fiber", new_top_area, simulator.blocks_area
	try:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg1, arg2))] )
	except:
		tmp_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index((aname, arg2, arg1))] )
	random.shuffle(tmp_actions)
	final_actions += tmp_actions
	return final_actions


def expert_demo_plan(simulator):
	'''
	use the most naive heuristic to create expert demo for planning: 
		remove all input blocks to table, and reassemble them according to goal
	'''
	input_stacks = simulator.input_stacks # each stack read from highest/top to lowest/bottom block, then filled by -1s
	goal_stacks = simulator.flipped_goal_stacks # each stack read from lowest/bottom to highest/top block, then filled by -1s
	final_actions = []
	# first need to parse input and goal
	action = "parse_input"
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	action = "parse_goal"
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	# remove everything from cur stacks
	table = [] # a cache storing blocks on the table
	cur_pointer = 0 # current pointer to cur stacks
	table_pointer = 0 # current pointer to table stacks
	for istack in range(simulator.puzzle_max_stacks):
		for jblock in range(simulator.stack_max_blocks): # from top/highest to bottom/lowest
			if input_stacks[istack][jblock]== -1: # no more blocks in this stack
				break
			action = "remove"
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			table.append(input_stacks[istack][jblock]) # record the blocks put on table
		if simulator.puzzle_max_stacks>1 and istack!=simulator.puzzle_max_stacks-1: # move to next stack if applicable
			action = "next_stack"
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			cur_pointer += 1
	# add blocks according to goal
	for istack in range(simulator.puzzle_max_stacks):
		for jblock in range(simulator.stack_max_blocks): # from bottom/lowest to top/highest
			if goal_stacks[istack][jblock] == -1: # no more blocks in this stack
				break # go to next goal stack
			blocktoadd = goal_stacks[istack][jblock]
			loc = table.index(blocktoadd) # table stack idx containing this block
			if loc - table_pointer > 0: # should move to the right to reach the table stack
				for _ in range(loc - table_pointer):
					action = "next_table"
					final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			elif loc - table_pointer < 0:
				for _ in range(table_pointer - loc):
					action = "previous_table"
					final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			table_pointer = loc # table pointer is now moved to the table stack to be added
			if istack - cur_pointer > 0: # cur stack pointer need to move to the right
				for _ in range(istack - cur_pointer):
					action = "next_stack"
					final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			elif istack - cur_pointer < 0: # cur stack pointer need to move to the left
				for _ in range(cur_pointer - istack):
					action = "previous_stack"
					final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			cur_pointer = istack # cur pointer moved to the stack to be added
			action = "add" # add the block 
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	return final_actions # list of integers representing actions



def _intersection(cur, goal):
	for i in range(len(cur)): # from bottom to top
		if cur[i]!=goal[i] or cur[i]==goal[i]==-1:
			ntoremove = sum([1 for b in cur[i:] if b!=-1 ])
			ntoadd =  sum([1 for b in goal[i:] if b!=-1])
			ncorrect = i
			return ntoremove, ntoadd, ncorrect
	return 0, 0, len(cur)


def oracle_demo_plan(simulator):
	'''
	optimal solution for planning: 
		remove non-intersected blocks, then reassemble 
	'''
	input_stacks = simulator.input_stacks # each stack read from highest/top to lowest/bottom block, then filled by -1s
	goal_stacks = simulator.flipped_goal_stacks # each stack read from lowest/bottom to highest/top block, then filled by -1s
	final_actions = []
	# first need to parse input and goal
	action = "parse_input"
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	action = "parse_goal"
	final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	# remove everything from cur stacks
	table = [] # a cache storing blocks on the table
	cur_pointer = 0 # current pointer to cur stacks
	table_pointer = 0 # current pointer to table stacks
	stacksntoremove = []
	stacksntoadd = []
	stacksncorrect = []
	for istack in range(simulator.puzzle_max_stacks):
		ntoremove, ntoadd, ncorrect = _intersection(simulator.flip(simulator.input_stacks)[istack], goal_stacks[istack])
		stacksntoremove.append(ntoremove)
		stacksntoadd.append(ntoadd)
		stacksncorrect.append(ncorrect)
		for jblock in range(ntoremove): # from top/highest to bottom/lowest
			assert input_stacks[istack][jblock]!=-1 # no more blocks in this stack
			action = "remove"
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			table.append(input_stacks[istack][jblock]) # record the blocks put on table
		if simulator.puzzle_max_stacks>1 and istack!=simulator.puzzle_max_stacks-1: # move to next stack if applicable
			if _intersection(simulator.flip(simulator.input_stacks)[istack+1], goal_stacks[istack+1])[0] > 0:
				action = "next_stack"
				final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
				cur_pointer += 1
	# add blocks according to goal
	while cur_pointer!=simulator.puzzle_max_stacks-1 and stacksntoadd[simulator.puzzle_max_stacks-1]>0: 
		action = "next_stack" # check if need to start from the last stack
		final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
		cur_pointer += 1
	for istack in range(cur_pointer, -1, -1):
		ntoadd = stacksntoadd[istack]
		ncorrect = stacksncorrect[istack]
		for jblock in range(ncorrect, ncorrect+ntoadd): # from bottom/lowest to top/highest
			assert goal_stacks[istack][jblock]!=-1 # should be nonempty
			blocktoadd = goal_stacks[istack][jblock]
			loc = table.index(blocktoadd) # table stack idx containing this block
			if loc - table_pointer > 0: # should move to the right to reach the table stack
				for _ in range(loc - table_pointer):
					action = "next_table"
					final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			elif loc - table_pointer < 0:
				for _ in range(table_pointer - loc):
					action = "previous_table"
					final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
			table_pointer = loc # table pointer is now moved to the table stack to be added
			action = "add" # add the block 
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
		if istack>0 and stacksntoadd[istack-1]>0:
			action = "previous_stack"
			final_actions.append( list(simulator.action_dict.keys())[list(simulator.action_dict.values()).index(action)] )
	return final_actions # list of integers representing actions



def sample_random_puzzle(puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
						puzzle_num_blocks, 
						curriculum, leak,
						compositional, compositional_type, compositional_eval, compositional_holdout):
	'''
	Create a random puzzle for blocksworld planning task.
	Input
		puzzle_max_stacks: (int) 
			max number of stacks possible in the initial/goal configuration
		puzzle_max_blocks: (int)
			max number of blocks possible in the initial/goal configuration
		stack_max_blocks: (int)
			max number of blocks in a stack
		puzzle_num_blocks: (2<=int<=puzzle_max_blocks, default None) 
			the specific number of blocks to have in the initial/goal configuration. 
			higher priority than curriculum
		curriculum: (0 or 2<=int<=puzzle_max_blocks, default None)
			current curriculum, which determines the distribution of total number of blocks in initial/goal config.
	Output
		puzzle_num_blocks: (int) 
			total number of blocks in the initial or goal config
		input_stacks: (list of list)
			initial configuration, each inner list is a stack representation, read from top/highest to bottom/lowest. 
		goal_stacks: (list of list)
	'''
	assert puzzle_num_blocks==None or (type(puzzle_num_blocks)==int and 2<=puzzle_num_blocks<=puzzle_max_blocks)
	while True:
		puzzle_num_blocks = random.choice(list(range(2, puzzle_max_blocks+1))) if (puzzle_num_blocks==None and curriculum==None) else puzzle_num_blocks
		if puzzle_num_blocks==None and curriculum!=None: # sample from curriculum
			assert 0==curriculum or (2 <= curriculum <= puzzle_max_blocks), f"should have 2 <= curriculum ({curriculum}) <= {puzzle_max_blocks}"
			if curriculum==0: # uniform
				puzzle_num_blocks = random.choice(list(range(2, puzzle_max_blocks+1)))
			else:
				if compositional and compositional_type=='newblock':
					curriculum = min(curriculum, puzzle_max_blocks-len(compositional_holdout))
				population = list(range(2, puzzle_max_blocks+1)) # possible number of blocks in puzzle
				weights = np.zeros_like(population, dtype=np.float32)
				if (not leak) or (compositional and compositional_type=='newblock'):
					weights[curriculum-2] += 0.7 # weight for current level
					weights[max(curriculum-3, 0)] += 0.15 # weight for the prev level
					weights[: max(curriculum-3, 1)] += 0.15 / max(curriculum-3, 1) # weight for easier levels. harder levels are 0
				else: # spoil harder levels beyond the current curriculum
					weights[curriculum-2] += 0.6 # weight for current level
					weights[max(curriculum-3, 0)] += 0.15 # weight for the prev level
					weights[: max(curriculum-3, 1)] += 0.15 / max(curriculum-3, 1) # easier levels
					weights[min(curriculum-1, len(population)-1):] += 0.1 / (len(population)-min(curriculum-1,len(population)-1)) # harder levels
				assert np.isclose(np.sum(weights), 1), f"weights {weights} should sum to 1, but have {np.sum(weights)}"
				puzzle_num_blocks = random.choices(population=population, weights=weights, k=1)[0]
		# sample input stacks
		available_blocks = list(range(puzzle_num_blocks)) # pool of all block ids for this puzzle
		if compositional and compositional_type=='newblock': # holdout a set of block ids
			if compositional_eval: # eval mode
				available_blocks = random.sample(compositional_holdout, # sample some blocks from holdout
										# k=min(random.choice(range(1,puzzle_num_blocks+1)), len(compositional_holdout)),
										k=1,
										)
				if len(available_blocks)<puzzle_num_blocks: # fill the rest blocks
					trainblocks = list(set(range(puzzle_max_blocks)) - set(compositional_holdout)) # blocks used for training
					unselectedblocks = list(set(range(puzzle_max_blocks)) - set(available_blocks)) # all remaining blocks including comp holdout
					# available_blocks += random.sample(unselectedblocks, k=puzzle_num_blocks-len(available_blocks))
					available_blocks += random.sample(trainblocks, k=puzzle_num_blocks-len(available_blocks))
					assert len(available_blocks) == puzzle_num_blocks
			else: # training mode
				available_blocks = list(set(range(puzzle_max_blocks)) - set(compositional_holdout))[:puzzle_num_blocks]
				assert len(available_blocks)>=puzzle_num_blocks, \
					f"compositional training {available_blocks} contains fewer elements than required by a puzzle ({puzzle_num_blocks})"
		input_blocks = copy.deepcopy(available_blocks)
		input_stacks = []
		for _ in range(puzzle_max_stacks):
			if len(available_blocks)==0:
				continue
			num_blocks_istack = random.choice(list(range( min(len(available_blocks), stack_max_blocks)+1 ))) 
			curstack = []
			for _ in range(num_blocks_istack): # sample block id one by one
				curblock = random.choice(available_blocks)
				curstack.append(curblock)
				available_blocks.remove(curblock)
			if curstack!=[]: # gathered some blocks for this stack
				input_stacks.append(curstack)
		if len(available_blocks) != 0: # remaining blocks from the pool will be added
			random.shuffle(available_blocks)
			istack = 0  # ith input stack to check
			while True: # spread the remaining blocks to all stacks of input
				istack = istack % puzzle_max_stacks 
				if len(input_stacks) <= istack: # not that many input stacks yet
					input_stacks.append([]) # add new input stack
				assert len(input_stacks[istack]) <= stack_max_blocks, \
					f"puzzle input stack {istack}: {input_stacks[istack]} has more than stack_max_blocks ({stack_max_blocks}) items"
				if len(input_stacks[istack]) < stack_max_blocks: # this input stack has spare space
					ab = available_blocks.pop() # add a remaining block
					input_stacks[istack].append(ab)
				istack += 1 # go to next input stack
				if len(available_blocks)==0:
					break # no more remaining blocks available, done
		# sample goal stacks
		available_blocks = input_blocks
		goal_stacks = []
		for _ in range(puzzle_max_stacks):
			if len(available_blocks)==0:
				continue
			num_blocks_istack = random.choice(list(range( min(len(available_blocks), stack_max_blocks)+1 ))) 
			curstack = []
			for _ in range(num_blocks_istack):
				curblock = random.choice(available_blocks)
				curstack.append(curblock)
				available_blocks.remove(curblock)
			if curstack!=[]:
				goal_stacks.append(curstack)
		if len(available_blocks) != 0: # remaining blocks will be added to the last stack
			random.shuffle(available_blocks)
			istack = 0  # ith input stack to check
			while True: # spread the remaining blocks to all stacks of input
				istack = istack % puzzle_max_stacks 
				if len(goal_stacks) <= istack: # not that many goal stacks yet
					goal_stacks.append([]) # add new input stack
				assert len(goal_stacks[istack]) <= stack_max_blocks, \
					f"puzzle goal stack {istack}: {goal_stacks[istack]} has more than stack_max_blocks ({stack_max_blocks}) items"
				if len(goal_stacks[istack]) < stack_max_blocks: # this input stack has spare space
					ab = available_blocks.pop() # add a remaining block
					goal_stacks[istack].append(ab)
				istack += 1 # go to next input stack
				if len(available_blocks)==0:
					break # no more remaining blocks available, done		
		assert (len(input_stacks) <= puzzle_max_stacks) and (len(goal_stacks) <= puzzle_max_stacks),\
			f"puzzle input {input_stacks} and goal {goal_stacks} should each have fewer than {puzzle_max_stacks} stacks"
		# found valid puzzle config, return
		if (input_stacks != goal_stacks):
			if ([input_stacks, goal_stacks] not in test_puzzles.test_puzzles):
				return puzzle_num_blocks, input_stacks, goal_stacks
			else:
				print(f"\n\nProposed puzzle [ {input_stacks}, {goal_stacks} ] overlaps with test puzzle, skip.\n\n")



