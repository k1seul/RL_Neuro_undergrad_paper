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
from GridWorld import GridWorld
import json 


def single_grid_train(rand_seed = 0, soft_max = False, action_mask_bool = False, fixed_trial = None): 
    gif_plotting = False # for Plotting gif path comparson data between agent and monkey 
    monkey_name = "p" 
    bool_pre_train = False
    env = GridWorld()

    random_agent = False

    game_name = "single_grid_dqn" if not(random_agent) else "single_grid_random_agent"



    
    hyperparameters_file_path = "hyperparameters/" + 'Vanilla_dqn' + ".json"



    with open(hyperparameters_file_path, "r") as file: 
        hyperparameters = json.load(file)

    data_dir, data_saver, writer = directory_setting(game_name, monkey_name,
        task_info=f"simple {game_name} training",
        run_num = rand_seed)
    
    ## Enviornment parameters 
    state_size = env.state_n
    action_size = env.action_n
   

    if not(soft_max):
        agent = agent_package.VanilaDqnAgent(state_size = state_size, 
                        action_size = action_size,
                        hyperparameters = hyperparameters,
                        seed_value=rand_seed,
                        )

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

    env.close()

    for trial_num in range(1000):

        if random_agent:
            agent.epsilon = 1
        

        if gif_plotting: 
            env = GridWorld(render_mode = "human", file_name = monkey_agent_compare_gif_dir + f"trial_{str(trial_num)}.gif")
        else:
            env = GridWorld() 


        trial_t = 0
 
        state, info = env.reset()

        reward = 0
        done = False 
        truncated = False 
        total_length = 1
        total_reward = 0 

        state_trajectories = []
        action_trajectories = [] 

        while not(done or truncated):
            trial_t += 1 

   

            action = agent.act(state, info)

            next_state, reward, done, truncated, info = env.step(action) 
            ## expected_reward = agent.get_expected_reward(state, action) 
            agent_reward = reward

            total_reward += reward 
            total_length += 1 

            agent.remember(state, action, agent_reward, next_state, done)

            

            agent.single_replay()

            state_trajectories.append(state)
            action_trajectories.append(action)
            state = next_state
            finished = done or truncated
            data_saver.record_visited_count(state = state, trial_num = trial_num, model = False)
            
            

            

            
        if done:
            agent.decay_epsilon()
        

        
        agent.save_network_weight(trial_num = trial_num)
        data_saver.save_visited_count()
        writer.add_scalar("total_reward", total_reward, trial_num)
        writer.add_scalar("total_length", total_length, trial_num)  
        writer.add_scalar("epsilon", agent.epsilon, trial_num)
        writer.add_scalar("reward_rate", total_reward/total_length, trial_num)
        data_saver.length_list.append(total_length)
        data_saver.reward_list.append(total_reward)
        data_saver.reward_rate_list.append(total_reward/total_length)



        os.system('cls' if os.name == 'nt' else 'clear')
        print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(trial_num, total_reward, agent.epsilon, total_length))
        if gif_plotting: 
            env.recrdr.save()
        env.close()

    writer.close() 
    data_saver.save_data() 


if __name__ == "__main__":
    single_grid_train()

        
