import os, sys, time, subprocess  
import numpy as np 
from MonkeyMazeEnv import MonkeyMazeEnv 
from MonkeyPath import MonkeyPath 
import agent_package
import path_analytic_tool
import plotting_functions 
from torch.utils.tensorboard import SummaryWriter
from pre_train import pre_train 
from directory_setting import directory_setting 
import json 

def direction_to_action(direction):
    if np.all(direction == np.array([1,0])):
        return 0
    elif np.all(direction == np.array([0,1])):
        return 1
    elif np.all(direction == np.array([-1,0])):
        return 2
    elif np.all(direction == np.array([0,-1])):
        return 3
    else:
        return 0


def monkey_evaluate(): 



    bool_pre_train = True
    env = MonkeyMazeEnv() 
    monkey_path = MonkeyPath(monkey_name='p')
    trial_length = monkey_path.trial_num
    trial_changed_start = [1,3,58,198,264,329,399,464]
    game_name = "monkey_evaluate"

    data_dir, data_saver, writer = directory_setting(game_name, monkey_name='p',
        task_info=f"simple {game_name} training with pretrain {bool_pre_train} ",
        run_num = 0)
    
    ## Enviornment parameters 
    state_size = env.state_n
    action_size = env.action_n
   
    for trial_num in range(trial_length):
 
        trial_start = monkey_path.get_start_position(trial_num) 
        trial_goal = monkey_path.get_goal_position(trial_num)
        trial_monkey_path = monkey_path.get_trial(trial_num = trial_num)

       

        env = MonkeyMazeEnv() 

        trial_path_max = len(trial_monkey_path)
        trial_t = 0
        env._monkey_location = trial_start
        state, info = env.reset(reward = trial_goal, start = trial_start)

        reward = 0
        done = False 
        truncated = False 
        total_length = 1
        total_reward = 0 

        state_trajectories = []
        action_trajectories = [] 

        while not(done or truncated):
            trial_t += 1 

            env._monkey_location = trial_monkey_path[min(trial_t, trial_path_max - 1)]
            

            action = direction_to_action(env._monkey_location - state)

            next_state, reward, done, truncated, info = env.step(action) 
            ## expected_reward = agent.get_expected_reward(state, action) 
            agent_reward = reward

            if trial_t >= trial_path_max +1:
                done = True

            total_reward += reward 
            total_length += 1 

            next_state = np.copy(env._monkey_location)

            state_trajectories.append(state)
            action_trajectories.append(action)
            state = next_state
            data_saver.record_visited_count(state = state, trial_num = trial_num, model = False)

            
            
        if total_reward < 6:
            total_reward += 8

       
        data_saver.save_visited_count()
        data_saver.length_list.append(total_length)
        data_saver.reward_list.append(total_reward)
        data_saver.reward_rate_list.append(total_reward/total_length)
        os.system('cls' if os.name == 'nt' else 'clear')
       
        env.close()

    writer.close() 
    data_saver.save_data() 


if __name__ == "__main__":
    monkey_evaluate()

        
