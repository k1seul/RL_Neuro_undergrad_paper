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


def simple_dqn_train(rand_seed = 0, soft_max = False, action_mask_bool = False, fixed_trial = None, bool_PER = False): 
    gif_plotting = False # for Plotting gif path comparson data between agent and monkey 
    monkey_name = "p" 
    bool_pre_train = True
    # Priortized Experience Replay 
    env = MonkeyMazeEnv() 
    monkey_path = MonkeyPath(monkey_name=monkey_name)
    trial_length = monkey_path.trial_num
    trial_changed_start = [1,3,58,198,264,329,399,464]
    if not(fixed_trial is None):
        fixed_trial = trial_changed_start[fixed_trial]
    game_name = "Vanilla_dqn" if not(soft_max) else "Softmax_dqn"
    game_name = game_name + f"_PER_{bool_PER}" if bool_PER else game_name



    
    hyperparameters_file_path = "hyperparameters/" + game_name + ".json"


    game_name = game_name + "_" + str(rand_seed)

    with open(hyperparameters_file_path, "r") as file: 
        hyperparameters = json.load(file)

    data_dir, data_saver, writer = directory_setting(game_name, monkey_name,
        task_info=f"simple {game_name} training with pretrain {bool_pre_train} and PER {bool_PER}",
        run_num = rand_seed)
    
    ## Enviornment parameters 
    state_size = env.state_n
    action_size = env.action_n
   

    if not(soft_max):
        agent = agent_package.VanilaDqnAgent(state_size = state_size, 
                        action_size = action_size,
                        hyperparameters = hyperparameters,
                        seed_value=rand_seed,
                        bool_PER=bool_PER)
        agent.action_mask_bool = action_mask_bool 
    else:
        """for softmax agent currently working on it """
        pass 

    agent.random_seed(rand_seed) 
    agent.weight_data_dir = data_dir + f"network_weight/"
    if not(os.path.exists(agent.weight_data_dir)):
        os.makedirs(agent.weight_data_dir)

    if bool_pre_train:
        pre_train(agent = agent, model = None, episode_num = 10)

    for trial_num in range(trial_length):
        if not(fixed_trial is None):
            saved_trial_num = trial_num
            trial_num = fixed_trial
        trial_start = monkey_path.get_start_position(trial_num) 
        trial_goal = monkey_path.get_goal_position(trial_num)
        trial_monkey_path = monkey_path.get_trial(trial_num = trial_num)
        if not(fixed_trial is None):
            trial_num = saved_trial_num
        monkey_agent_compare_gif_dir = data_dir + f"monkey_compare_agent/"
        if not(os.path.exists(monkey_agent_compare_gif_dir)):
            os.makedirs(monkey_agent_compare_gif_dir)

        if gif_plotting: 
            env = MonkeyMazeEnv(render_mode = "human", file_name = monkey_agent_compare_gif_dir + f"trial_{str(trial_num)}.gif")
        else:
            env = MonkeyMazeEnv() 

        #env.no_small_reward = True 

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

            action = agent.act(state, info)

            next_state, reward, done, truncated, info = env.step(action) 
            ## expected_reward = agent.get_expected_reward(state, action) 
            agent_reward = reward

            total_reward += reward 
            total_length += 1 

            agent.remember(state, action, agent_reward, next_state, done)

            

            agent.replay(TD_sample=bool_PER)

            state_trajectories.append(state)
            action_trajectories.append(action)
            state = next_state
            finished = done or truncated
            data_saver.record_visited_count(state = state, trial_num = trial_num, model = False)

            plotting_functions.plot_all_functions(agent=agent, model=None, i_episode = trial_num, trial_t = trial_t, 
                                                state = state, reward_location=trial_goal, data_dir=data_dir, finished=finished)

            
        if done:
            agent.decay_epsilon()
        
        while trial_t < trial_path_max - 1:
            trial_t += 1 
            env._monkey_location = trial_monkey_path[trial_t]
            env._render_frame() 

        path_analytic_tool.get_all_path_analytic_out(agent = agent, agent_path = state_trajectories, agent_action_seq = action_trajectories,
                monkey_path=trial_monkey_path, writer=writer, trial_num = trial_num, total_reward = total_reward, 
                total_length = total_length, data_saver = data_saver) 
        agent.save_network_weight(trial_num = trial_num)
        data_saver.save_visited_count()
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(trial_num, total_reward, agent.epsilon, total_length))
        if gif_plotting: 
            env.recrdr.save()
        env.close()

    writer.close() 
    data_saver.save_data() 


if __name__ == "__main__":
    simple_dqn_train(rand_seed = 0,bool_PER=True, fixed_trial=0)

        
