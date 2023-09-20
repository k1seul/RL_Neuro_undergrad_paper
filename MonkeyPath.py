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

        start_end_location = []
        for i in range(self.trial_num):
            start_end_location.append(tuple(self.pos_sequence[i][0][0]))

        start_end_location = list(set(start_end_location))
        self.start_end_locations = start_end_location

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
            end_locations.append(tuple(self.pos_sequence[i][-1][0]))

        end_locations = list(set(end_locations))
        return end_locations

    def location2index(self, location):
        index_num = self.start_end_locations.index(tuple(location))
        return index_num


if __name__ == "__main__":
    monkey_path = MonkeyPath()
    print(monkey_path.get_trial(0))
