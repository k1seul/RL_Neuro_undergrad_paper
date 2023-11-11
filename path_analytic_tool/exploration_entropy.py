import numpy as np 
import sys , os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from MonkeyPath import MonkeyPath


def exploration_entropy(path):
    state_counter = {}
    total_cell_num = 52 
    for state in path:
        if not(tuple(state) in state_counter):
            state_counter[tuple(state)] = 0
        state_counter[tuple(state)] += 1

    state_counter = np.array(list(state_counter.values()))
    state_p = state_counter/sum(state_counter)
    entropy = -sum(state_p*np.log(state_p))

    visited_cell_num = len(state_counter)
    explore_map_percentage = visited_cell_num/total_cell_num * 100 

    return entropy, explore_map_percentage


if __name__ == "__main__":
    import pickle
    
    MkPath = MonkeyPath()
    explore_percentage_list = []
    entropy_list = [] 
    for i in range(519):
        entropy = exploration_entropy(MkPath.get_trial(i))
        explore_percentage_list.append(entropy[1])
        entropy_list.append(entropy[0])

    explore_extropy_dict = {"explore_percentage":explore_percentage_list, "entropy":entropy_list}

    with open("explore_extropy_dict.pkl", "wb") as f:
        pickle.dump(explore_extropy_dict, f)
