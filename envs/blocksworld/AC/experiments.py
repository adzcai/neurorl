from envs.blocksworld.AC.bw_apps import *
# import numpy as np
from scipy.stats import sem
from envs.blocksworld.parse import Simulator
from envs.blocksworld import utils
		
class SyntheticBrain(Simulator):
	def __init__(self,
			  puzzle_max_blocks, 
			  nblocks,
			  verbose=False):
		super().__init__(puzzle_max_blocks=puzzle_max_blocks, 
						stack_max_blocks=nblocks,
						verbose=verbose)
		self.num_blocks = nblocks

	def parse(self, goal):
		self.reset(shuffle=False, difficulty_mode=self.num_blocks, cur_curriculum_level=None)
		assert self.num_blocks == len(goal), f"self.num_blocks{self.num_blocks} should be equal to len(goal) {len(goal)}"
		self.goal = [None] * self.stack_max_blocks
		self.goal[:self.num_blocks] = goal
		actions = utils.expert_demo_parse(self.goal, self.num_blocks)
		for a in actions:
			self.step(a)
		readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, len(self.goal), self.blocks_area)
		return readout

def parse_recall_performance(maxlen, puzzle_max_blocks, neans=[1e4], nrepeats=1):
	# ratio of item recalled as a function of parse chain length
	assert puzzle_max_blocks >= maxlen, f"puzzle_max_blocks {puzzle_max_blocks} should be >= maxlen {maxlen}"
	prefix = "G"
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	ac_accuracy = []
	ac_accuracy_sem = []
	synth_accuracy = []
	synth_accuracy_sem = []
	for i, nean in enumerate(neans):
		print(f"nean={nean}")
		nean_accuracy = []
		nean_accuracy_sem = []
		for nitems in range(1, maxlen+1):
			print(f"chaining {nitems} items...")
			stacks = [list(range(nitems))]
			ac_acc = []
			synth_acc = []
			for irepeat in range(nrepeats):
				bb = BlocksBrain(blocks_number=puzzle_max_blocks, other_areas=oa, p=0.1, eak=int(np.sqrt(nean)), nean=nean, neak=int(np.sqrt(nean)), db=0.2)	
				parse(bb, stacks=stacks, prefix=prefix)
				r = readout(bb, stacks_number=len(stacks), stacks_lengths=[nitems], top_areas=[0], prefix=prefix)
				ac_acc.append(_ratio_matched(r[0], stacks[0]))
				if i==0: # synthetic parse
					sb = SyntheticBrain(puzzle_max_blocks=puzzle_max_blocks, nblocks=nitems)
					sr = sb.parse(stacks[0])
					synth_acc.append(_ratio_matched(sr, stacks[0]))
			nean_accuracy.append(round(np.mean(ac_acc), 6))
			nean_accuracy_sem.append(round(sem(ac_acc), 6))
			if i==0:
				print("synthetic parsing...")
				synth_accuracy.append(round(np.mean(synth_acc), 6))
				synth_accuracy_sem.append(round(sem(synth_acc), 6))
		ac_accuracy.append(nean_accuracy)
		ac_accuracy_sem.append(nean_accuracy_sem)
	print(f"parse_lengths={list(range(1,maxlen+1))}\
			\nneans={neans}\
			\nac_accuracy={ac_accuracy}\
			\nac_accuracy_sem={ac_accuracy_sem}\
			\nsynth_accuracy={synth_accuracy}\
			\nsynth_accuracy_sem={synth_accuracy_sem}")


def _ratio_matched(stack, target):
	# proportion of items in stack that match the target items
	assert len(stack)==len(target), f"stack len {len(stack)} and target len {len(target)} should match"
	nmatch = 0
	for i in range(len(target)):
		if stack[i]==None:
			continue
		if int(stack[i])==int(target[i]):
			nmatch += 1
	return nmatch / len(target)


'''
salloc -p test -t 0-01:00 --mem=600000 
module load python/3.10.12-fasrc01
mamba activate neurorl
'''

if __name__ == "__main__":
	random.seed(0)
	parse_recall_performance(
							maxlen=9,
							puzzle_max_blocks=20,
							neans=[1e3,1e4,1e5],
							nrepeats=3,
							)