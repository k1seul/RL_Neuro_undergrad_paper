import numpy as np
import pickle 
import os 

class DataSaver():
    """ for saving tensorboard output data to numpy array"""
    def __init__(self, data_dir = "data/", model_name="vanilla_dqn", run_num=0):
        self.data_dir = data_dir 
        self.file_name = data_dir + model_name + f"_{str(run_num)}.pkl" 

        self.length_list = [] 
        self.reward_list = [] 
        self.reward_rate_list = [] 
        self.shannon_value_list = []
        self.path_sim_list = [] 
        self.choice_sim_percent_list = [] 
        self.monkey_max_choice_compare_percent_list = [] 
        self.exploration_entropy_list = [] 
        self.explore_percentage_list = [] 
        self.agent_trajectory_list = [] 
        self.choice_sim_percent_agent_self = [] 
        self.agent_visited_count = []
        self.model_visited_count = []
        self.agent_counter = np.zeros([11,11])
        self.model_counter = np.zeros([11,11])
        self.trial_num_memory = -1

    def record_visited_count(self, state, trial_num, model=False, curiosity=False):
        if not(self.trial_num_memory == trial_num):
            self.agent_counter = np.zeros([11,11])
            self.model_counter = np.zeros([11,11])
            self.curiosity_counter = np.zeros([11,11])
            self.trial_num_memory = trial_num
        
        if model:
            if not(curiosity):
                self.model_counter[state[0], state[1]] += 1
            else:
                self.curiosity_counter[state[0], state[1]] += 1
        else:
            self.agent_counter[state[0], state[1]] += 1
    def save_visited_count(self):
        self.agent_visited_count.append(self.agent_counter)
        self.model_visited_count.append([self.model_counter, self.curiosity_counter])
        

        


    

    def record_data(self, length, reward, reward_rate, shannon_value, path_sim, choice_sim_percent, monkey_max_choice_compare_percent, exploration_entropy, explore_percentage, trajectory,
                    choice_sim_percent_agent_self):
       
        self.length_list.append(length)
        self.reward_list.append(reward)
        self.reward_rate_list.append(reward_rate)
        self.shannon_value_list.append(shannon_value)
        self.path_sim_list.append(path_sim)
        self.choice_sim_percent_list.append(choice_sim_percent)
        self.monkey_max_choice_compare_percent_list.append(monkey_max_choice_compare_percent)
        self.exploration_entropy_list.append(exploration_entropy)
        self.explore_percentage_list.append(explore_percentage)
        self.agent_trajectory_list.append(trajectory)
        self.choice_sim_percent_agent_self.append(choice_sim_percent_agent_self)

    def save_data(self):
        data = {
            "length": self.length_list,
            "reward": self.reward_list,
            "reward_rate": self.reward_rate_list,
            "shannon_value": self.shannon_value_list,
            "path_sim": self.path_sim_list,
            "choice_sim_percent": self.choice_sim_percent_list,
            "monkey_max_choice_compare_percent": self.monkey_max_choice_compare_percent_list,
            "exploration_entropy": self.exploration_entropy_list,
            "explore_percentage": self.explore_percentage_list,
            "agent_trajectory": self.agent_trajectory_list, 
            "choice_sim_percent_agent_self": self.choice_sim_percent_agent_self,
            "agent_visited_count": self.agent_visited_count,
            "model_visited_count": self.model_visited_count
        }

        if not(os.path.exists(self.data_dir)):
            os.makedirs(self.data_dir)

        with open(self.file_name, "wb") as f:
            pickle.dump(data, f)
