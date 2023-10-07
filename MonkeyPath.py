import numpy as np
import scipy.io


class MonkeyPath:
    """ "For getting Monkey grid positional data, use get_~ to get the start of goal position of given trial_num
    and use get_trial for sequence of certain trial"""

    def __init__(self, monkey_name="p"):
        mat_file_name = f"matlab_data/pos_seq_{monkey_name}"
        mat_file = scipy.io.loadmat(mat_file_name)
        pos_sequence = mat_file["pos_sequence_all"]
        self.pos_sequence = pos_sequence
        self.trial_num = pos_sequence.shape[0]
        self.location_num = 16
        self.goal_index = [] 

        start_end_location = []
        for i in range(self.trial_num):
            start_end_location.append(tuple(self.pos_sequence[i][0][0]))

        start_end_location = list(set(start_end_location))
        self.start_end_locations = start_end_location
        self.goal_list = self.end_locations()
        self.get_all_index()

    def get_goal_position(self, trial_num):
        return self.pos_sequence[trial_num][0][-1]

    def get_start_position(self, trial_num):
        return self.pos_sequence[trial_num][0][0]

    def get_trial(self, trial_num):
        return self.pos_sequence[trial_num][0]

    def start_locations(self):
        starting_locations = []
        for i in range(self.trial_num):
            starting_locations.append(tuple(self.pos_sequence[i][0][0]))
        starting_locations = set(starting_locations)
        starting_locations = list(starting_locations)

        return starting_locations

    def end_locations(self):
        end_locations = []
        for i in range(self.trial_num):
            if not(tuple(self.pos_sequence[i][0][-1]) in end_locations):
                end_locations.append(tuple(self.pos_sequence[i][0][-1]))
            else:
                pass


        end_locations = list(end_locations)
        return end_locations
    

    
    

    def location2index(self, location):
        index_num = self.goal_list.index(tuple(location))
        return index_num
    
    def get_all_index(self):
        for i in range(self.trial_num):
            self.goal_index.append(self.location2index(self.get_goal_position(i)))


if __name__ == "__main__":
    monkey_path = MonkeyPath()
    print(monkey_path.get_trial(0))
    print(monkey_path.get_goal_position(0))
    print(monkey_path.end_locations())
    print(monkey_path.goal_index)
