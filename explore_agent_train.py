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


def explore_dqn_train(rand_seed = 0, soft_max = False, action_mask_bool = False, fixed_trial = None,
                      log_weight=1, log_bias = 0.5): 
    gif_plotting = False # for Plotting gif path comparson data between agent and monkey 
    monkey_name = "p" 
    bool_PER = False # Priortized Experience Replay 
    env = MonkeyMazeEnv() 
    monkey_path = MonkeyPath(monkey_name=monkey_name)
    trial_length = monkey_path.trial_num
    trial_changed_start = [1,3,58,198,264,329,399,464]
    game_name = "Explore_dqn"


    
    hyperparameters_file_path = "hyperparameters/" + game_name + ".json"


    game_name = game_name + f"_log_weight_{log_weight}_log_bias_{log_bias}_PER_{bool_PER}"

    with open(hyperparameters_file_path, "r") as file: 
        hyperparameters = json.load(file)

    data_dir, data_saver, writer = directory_setting(game_name, monkey_name,
        task_info=f"simple {game_name} training with PER {bool_PER}",
        run_num = rand_seed)
    
    ## Enviornment parameters 
    state_size = env.state_n
    action_size = env.action_n
   

    if not(soft_max):
        agent = agent_package.ExploreDqnAgent(state_size = state_size, 
                        action_size = action_size,
                        hyperparameters = hyperparameters,
                        seed_value=rand_seed,
                        log_weight=log_weight,
                        log_bias=log_bias)
        agent.action_mask_bool = action_mask_bool 
    else:
        """for softmax agent currently working on it """
        pass 

    agent.random_seed(rand_seed) 
    agent.weight_data_dir = data_dir + f"network_weight/"
    if not(os.path.exists(agent.weight_data_dir)):
        os.makedirs(agent.weight_data_dir)

    env = MonkeyMazeEnv(no_reward=True)

    for trial_num in range(trial_length):
        



        trial_t = 0
        state, info = env.reset()
        agent.reset_counter()

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
            agent.update_counter(next_state)
            internal_reward = agent.calculate_explore_reward(next_state)

            total_reward += reward 
            total_length += 1 

            reward = reward + internal_reward
            print(internal_reward)

            done = agent.calculate_visited_num()

            agent.remember(state, action, reward, next_state, done)

            

            agent.replay(TD_sample=bool_PER)

            state_trajectories.append(state)
            action_trajectories.append(action)
            state = next_state
            finished = done or truncated
            data_saver.record_visited_count(state = state, trial_num = trial_num, model = False)

            plotting_functions.plot_all_functions(agent=agent, model=None, i_episode = trial_num, trial_t = trial_t, 
                                                state = state, reward_location=np.array([0,0]), data_dir=data_dir, finished=finished)

            
        if done:
            agent.decay_epsilon()

        path_analytic_tool.get_all_path_analytic_out(agent = agent, agent_path = state_trajectories, agent_action_seq = action_trajectories,
                monkey_path=[], writer=writer, trial_num = trial_num, total_reward = total_reward, 
                total_length = total_length, data_saver = data_saver, no_monkey = True)
        agent.save_network_weight(trial_num = trial_num)
        data_saver.save_visited_count()

        print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(trial_num, total_reward, agent.epsilon, total_length))
        if gif_plotting: 
            env.recrdr.save()
        env.close()

    writer.close() 
    data_saver.save_data() 


if __name__ == "__main__":
    explore_dqn_train(rand_seed = 0, soft_max = False, action_mask_bool = False, fixed_trial = None)
