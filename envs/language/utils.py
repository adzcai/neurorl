import random
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

# Fixed area stats for explicit areas
LEX_SIZE = 20

# Actions
DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"
# Skip firing in this round, just activate the word in LEX/DET/other word areas.
# All other rules for these lexical items should be in PRE_RULES.
ACTIVATE_ONLY = "ACTIVATE_ONLY"
CLEAR_DET = "CLEAR_DET"

AREAS = [LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]
EXPLICIT_AREAS = [LEX]
RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P]

def init_simulator_areas():
	# initialize all areas, head
	pass

def parse_noun():
	# actions to parse a generic noun
	pass

def parse_transverb():
	pass

def parse_intransverb():
	pass

def parse_copula():
	pass

def parse_adverb():
	pass

def parse_det():
	pass

def parse_adj():
	pass

def parse_prep():
	pass


def expert_demo_language(simulator):
	pass

def synthetic_readout(simulator):
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
	pass

def calculate_unit_reward():
	pass

def calculate_readout_reward():
	pass

def top():
	pass

def is_last_block():
	pass

def all_fiber_closed():
	pass

def sample_sentence():
	pass

def create_episode(shuffle, difficulty_mode, cur_curriculum_level):
	'''
	Create a goal stack confirguation for the episode.
	Input
		shuffle: (boolean)
			whether or not to shuffle items in the goal stack
		difficulty_mode: {'curriculum', 'uniform', 0, 1<=int<=stack_max_blocks}
			determines the number of blocks in goal stack
			if 'curriculum', sample a number from a curriculum distribution
			if 'uniform' or 0, sample a number from uniform distribution [1,stack_max_blocks]
			if 1<=int<=stack_max_blocks: number of blocks in goal will be equal to this given number
		cur_curriculum_level: {None, 1<=int<=stack_max_blocks}
			only needed when difficulty_mode=='curriculum', otherwise None
			specify the current number of blocks to focus on for the curriculum
	'''
	goal = [None] * self.stack_max_blocks # dummy goal template, to be filled
	num_blocks = None # actual number of blocks in the stack, to be modified
	if difficulty_mode=='curriculum':
		assert cur_curriculum_level!=None, f"requested curriculum but current level is not given"
		num_blocks = self.__sample_from_curriculum(cur_curriculum_level)
	elif difficulty_mode=='uniform' or (type(difficulty_mode)==int and difficulty_mode==0): # uniform mode
		num_blocks = random.randint(1, self.stack_max_blocks)
	elif type(difficulty_mode)==bool and (difficulty_mode==False): # default max blocks to parse
		num_blocks = self.stack_max_blocks
	elif type(difficulty_mode)==int:
		assert 1<=difficulty_mode<=self.stack_max_blocks, \
			f"invalid difficulty mode: {difficulty_mode}, should be 'curriculum', or 0, or values in [1, {self.stack_max_blocks}]"
		num_blocks = difficulty_mode
	else:
		raise ValueError(f"unrecognized difficulty mode {difficulty_mode} (type {type(difficulty_mode)})")
	assert num_blocks <= self.stack_max_blocks, \
		f"number of actual blocks to parse {num_blocks} should be smaller than stack_max_blocks {self.stack_max_blocks}"
	stack = random.sample(list(range(self.max_lexicon)), num_blocks) # the actual blocks in the stack
	if shuffle:
		random.shuffle(stack)
	goal[:num_blocks] = stack
	return num_blocks, goal

def sample_from_curriculum(cur_curriculum_level):
	'''
	randomly sample a number of blocks for the goal stack, based on the given curriculum
	Input:	
		cur_curriculum_level: (1<=int<=stack_max_blocks)
			the curriculum distribution will be [r, r, ..., 0.15, 0.7, 0, 0, ...]
			where 0.7 is probability for the currently focused number of blocks for the curriculum
			0.15 for the immediate previous curriculum
			higher levels will have 0 probability
			lower levels will have uniform probability given whatever prob is left
	Return:
		num_blocks: (int)
			the number of blocks in goal for this episode
	'''
	assert 1 <= cur_curriculum_level <= self.stack_max_blocks, f"should have 1<= cur_curriculum_level ({cur_curriculum_level}) <= {self.stack_max_blocks}"
	population = list(range(1, self.stack_max_blocks+1)) # possible number of blocks
	weights = np.zeros(self.stack_max_blocks)
	weights[cur_curriculum_level-1] += 0.7 # weight for current level
	weights[max(cur_curriculum_level-2, 0)] += 0.15 # weight for the prev level
	weights[: max(cur_curriculum_level-2, 1)] += 0.15 / max(cur_curriculum_level-2, 1)
	assert np.sum(weights)==1, f"weights {weights} should sum to 1"
	num_blocks = random.choices(population=population, weights=weights, k=1)[0]
	return num_blocks