import random
import numpy as np
import pprint
import envs.blocksworld.utils as bwutils
from envs.blocksworld.plan import Simulator

def train_prompt(nsamples, 
				puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
				puzzle_num_blocks, 
				curriculum, leak,
				compositional, compositional_type, compositional_eval, compositional_holdout):
	text = "Here are some example puzzles and the corresponding actions to solve the puzzles:\n"
	for isample in range(nsamples):
		_, input_stacks, goal_stacks = bwutils.sample_random_puzzle(puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
							puzzle_num_blocks, 
							curriculum, leak,
							compositional, compositional_type, compositional_eval, compositional_holdout)
		text += f"Input stacks: {input_stacks}, goal stacks: {goal_stacks}.\n"
		sim = Simulator(puzzle_max_stacks=puzzle_max_stacks, 
						puzzle_max_blocks=puzzle_max_blocks, 
						stack_max_blocks=stack_max_blocks, 
						evaluation=True, eval_puzzle_num_blocks=None,
						compositional=False, compositional_type=None, compositional_holdout=None,
						test_puzzle=[input_stacks, goal_stacks])
		sim.reset()
		action_dict = sim.action_dict
		expert_demo = bwutils.expert_demo_plan(sim)
		text += "Actions: ["
		for i, a in enumerate(expert_demo):
			if i==0:
				text += str(action_dict[a])
			else:
				text += ", "
				text += str(action_dict[a])
		text += "]\n\n"
	return text


def test_prompt(nsamples, 
				puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
				puzzle_num_blocks, 
				curriculum, leak,
				compositional, compositional_type, compositional_eval, compositional_holdout):
	puzzles = []
	text = "\nHere are the test puzzles, please produce the corresponding actions to solve each of these puzzles:\n"
	answers = "\nAnswers:\n"
	for isample in range(nsamples):
		_, input_stacks, goal_stacks = bwutils.sample_random_puzzle(puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
							puzzle_num_blocks, 
							curriculum, leak,
							compositional, compositional_type, compositional_eval, compositional_holdout)
		text += f"Input stacks: {input_stacks}, goal stacks: {goal_stacks}.\n"
		puzzles.append([input_stacks, goal_stacks])
		sim = Simulator(puzzle_max_stacks=puzzle_max_stacks, 
						puzzle_max_blocks=puzzle_max_blocks, 
						stack_max_blocks=stack_max_blocks, 
						evaluation=True, eval_puzzle_num_blocks=None,
						compositional=False, compositional_type=None, compositional_holdout=None,
						test_puzzle=[input_stacks, goal_stacks])
		sim.reset()
		action_dict = sim.action_dict
		expert_demo = bwutils.expert_demo_plan(sim)
		answers += f"[{input_stacks}, {goal_stacks}]\n"
		answers += "["
		for i, a in enumerate(expert_demo):
			if i==0:
				answers += str(action_dict[a])
			else:
				answers += ", "
				answers += str(action_dict[a])
		answers += "]\n\n"
	text += "\nPlease produce the sequence of actions for each test puzzle, following the same format as the examples above.\n\
Do not add explanation. Only produce the lists of actions."
	return text, answers, puzzles

def accuracy(nsamples, puzzles, actions,
				puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
				puzzle_num_blocks, 
				curriculum, leak,
				compositional, compositional_type, compositional_eval, compositional_holdout):
	solved = []
	rewards = []
	for isample in range(nsamples):
		input_stacks, goal_stacks = puzzles[isample]
		sim = Simulator(puzzle_max_stacks=puzzle_max_stacks, 
						puzzle_max_blocks=puzzle_max_blocks, 
						stack_max_blocks=stack_max_blocks, 
						evaluation=True, eval_puzzle_num_blocks=None,
						compositional=False, compositional_type=None, compositional_holdout=None,
						test_puzzle=[input_stacks, goal_stacks])
		sim.reset()
		action_dict = sim.action_dict
		er = 0
		truncated, terminated = False, False
		for i, a in enumerate(actions[isample]):
			aidx = list(action_dict.values()).index(a)
			state, reward, terminated, truncated, info = sim.step(aidx)
			er += reward
			if truncated or terminated:
				break
		if truncated:
			solved.append(0)
		elif terminated:
			solved.append(1)
		else:
			solved.append(0)
		rewards.append(er)
	print(np.mean(solved), solved)
	print(np.mean(rewards), rewards)
	return solved, rewards

def main(ntrain, ntest, 
				puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
				puzzle_num_blocks, 
				curriculum, leak,
				compositional, compositional_type, compositional_eval, compositional_holdout):
	instructions = "I will now teach you how to solve Blocks-World puzzles.\n\
Each puzzle consists of an input stacks configuration, and a goal stacks configuration.\n\
Each stack contains 1 to several blocks.\n\
Your goal is to produce a sequence of actions to match the goal configuration, starting from the input configuration.\n\
You may use the table to store blocks temporarily. \n\
The table has unlimited capacity and does not need to form stacks. That means, all blocks can lay flat on the table, if necessary.\n\
Each block has its unique identity, represented by its id number (an integer).\n\
Each stack is represented by a list of blocks (integers).\n\
The first element in the list corresponds to the top block in the stack, \
and the last element in the list corresponds to the bottom block in the stack.\n\
I will show you some examples for how to solve the puzzles.\n\
Then you will be asked to solve some other puzzles for me.\n"
	train = train_prompt(ntrain, 
				puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
				puzzle_num_blocks, 
				curriculum, leak,
				compositional, compositional_type, compositional_eval, compositional_holdout)
	test, answers, puzzles = test_prompt(ntest, 
				puzzle_max_stacks, puzzle_max_blocks, stack_max_blocks,
				puzzle_num_blocks, 
				curriculum, leak,
				compositional, compositional_type, compositional_eval, compositional_holdout)
	final = instructions + train + test
	print(final)
	print(answers)
	return puzzles


if __name__ == "__main__":
	random.seed(0)
	test_puzzles = main(ntrain=20, ntest=5, 
			puzzle_max_stacks=5, puzzle_max_blocks=10, stack_max_blocks=7,
			puzzle_num_blocks=None, 
			curriculum=4, leak=False,
			compositional=False, compositional_type=None, compositional_eval=False, compositional_holdout=None)
# 	actions = [["parse_input", "parse_goal", "remove", "remove", "remove", "remove", "next_stack", "remove", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "next_table", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "next_table", "next_stack", "add", "previous_table", "next_table", "next_stack", "add", "previous_table", "add"],
# ["parse_input", "parse_goal", "remove", "next_stack", "remove", "remove", "next_stack", "remove", "remove", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "previous_table", "next_stack", "add", "previous_table", "add", "previous_table", "next_stack", "add"],
# ["parse_input", "parse_goal", "remove", "remove", "remove", "next_stack", "remove", "remove", "next_stack", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "next_table", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "previous_table", "next_stack", "add", "next_table", "add", "next_stack", "next_stack", "add", "previous_table", "next_stack", "add"],
# ["parse_input", "parse_goal", "remove", "remove", "remove", "remove", "remove", "next_stack", "next_stack", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "next_table", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "next_table", "next_stack", "add", "next_table", "next_table", "next_stack", "add", "previous_table", "add", "previous_table", "add"],
# ["parse_input", "parse_goal", "remove", "remove", "remove", "remove", "remove", "next_stack", "next_stack", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "previous_table", "add", "previous_table", "next_stack", "add", "next_table", "next_table", "next_stack", "add"],]
	actions = [
["parse_input", "parse_goal", "remove", "remove", "next_stack", "remove", "next_stack", "next_stack", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "next_table", "add", "next_table", "next_table", "next_stack", "add"],

["parse_input", "parse_goal", "remove", "remove", "remove", "remove", "next_stack", "next_stack", "next_stack", "next_stack", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "next_table", "add", "previous_table", "add", "next_table", "next_stack", "add"],

["parse_input", "parse_goal", "remove", "next_stack", "remove", "remove", "next_stack", "remove", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "next_table", "previous_table", "add", "previous_table", "add", "next_table", "next_stack", "add"],

["parse_input", "parse_goal", "remove", "remove", "next_stack", "remove", "remove", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "previous_table", "previous_table", "add", "previous_table", "next_stack", "add", "next_table", "add"],

["parse_input", "parse_goal", "remove", "remove", "remove", "next_stack", "remove", "next_stack", "next_stack", "next_stack", "next_table", "next_table", "next_table", "previous_stack", "previous_stack", "previous_stack", "previous_stack", "add", "previous_table", "add", "previous_table", "add", "next_table", "next_stack", "add"],]
	accuracy(nsamples=5, puzzles=test_puzzles, actions=actions,
			puzzle_max_stacks=5, puzzle_max_blocks=10, stack_max_blocks=7,
			puzzle_num_blocks=None, 
			curriculum=4, leak=False,
			compositional=False, compositional_type=None, compositional_eval=False, compositional_holdout=None)