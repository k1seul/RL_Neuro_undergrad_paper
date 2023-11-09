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



def curiosity_based_train(rand_seed = 0, maxent=True, simulation_num = 3, simulation_max_episode = 20):

    gif_plotting = False      
    monkey_name = "p" 
    bool_pre_train = True 
    bool_PER = False 
    env = MonkeyMazeEnv() 
    monkey_path = MonkeyPath(monkey_name=monkey_name)
    trial_length = monkey_path.trial_num
    game_name = "Curiosity_counter_model" if not(maxent) else "Curiosity_maxent_model"

    hyperparameters_file_path = "hyperparameters/" + "Curiosity_counter_model" + ".json"
    with open(hyperparameters_file_path, "r") as file: 
        hyperparameters = json.load(file)
    data_dir, data_saver, writer = directory_setting(game_name, monkey_name,
        task_info=f"simple {game_name} training with pretrain {bool_pre_train} and PER {bool_PER}",
        run_num = rand_seed)
    
    game_name = game_name + "_" + str(rand_seed)
    ## agent_setting 
    state_size = env.state_n
    action_size = env.action_n
    

    agent = agent_package.CuriosityBasedAgent(state_size = state_size,
                                            action_size = action_size,
                                            hyperparameters=hyperparameters,
                                            explore_agent = True) 
    
    model = agent_package.CuriosityCounterModelTable(data_dir = data_dir)
    model.change_simulation_num(simulation_num = simulation_num, simulation_max_episode = simulation_max_episode)

    agent.random_seed(rand_seed)
    model.random_seed(rand_seed) 
    agent.weight_data_dir = data_dir + f"network_weight/"
    if not(os.path.exists(agent.weight_data_dir)):
        os.makedirs(agent.weight_data_dir)

    if bool_pre_train:
        pre_train(agent=agent, model=model) 

    for trial_num in range(trial_length):
        trial_start = monkey_path.get_start_position(trial_num) 
        trial_goal = monkey_path.get_goal_position(trial_num)
        trial_monkey_path = monkey_path.get_trial(trial_num = trial_num)
        monkey_agent_compare_gif_dir = data_dir + f"monkey_compare_agent/"
        if not(os.path.exists(monkey_agent_compare_gif_dir)):
            os.makedirs(monkey_agent_compare_gif_dir)

        if gif_plotting: 
            env = MonkeyMazeEnv(render_mode = "human", file_name = monkey_agent_compare_gif_dir + f"trial_{str(trial_num)}.gif")
        else:
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

        model.reset()
        model.t = 0 
        model.draw_map(episode_num = trial_num)
        model.record_map(state = state, reward = reward, done = False, i_episode = trial_num)

        state_trajectories = []
        action_trajectories = [] 

        while not(done or truncated):
            trial_t += 1 

            env._monkey_location = trial_monkey_path[min(trial_t, trial_path_max - 1)]

            action = agent.act(state)

            next_state, reward, done, truncated, info = env.step(action) 

            total_reward += reward 
            total_length += 1 
            agent.remember(state, action, reward, next_state, done)
            model.record_map(state = next_state , reward = reward, done = done, i_episode = trial_num)

            

            agent.replay(TD_sample=bool_PER)
            

            state_trajectories.append(state)
            action_trajectories.append(action)
            state = next_state
            finished = done or truncated
            data_saver.record_visited_count(state = state, trial_num = trial_num, model = False)


            reward_known = model.known_reward() 

            if reward_known:
                model.model_simulate(agent = agent, state = state, reset = True,trial_num=trial_num, data_saver=data_saver)
                model.curiosity_simulate(agent, state, trial_num=trial_num, data_saver=data_saver)
            else: 
                model.curiosity_simulate(agent, state, exploit_update=True, maxent_update=maxent, trial_num=trial_num, data_saver=data_saver)
        

            plotting_functions.plot_all_functions(agent = agent, model = model, i_episode = trial_num,
                                                  trial_t = trial_t, state = state,
                                                  reward_location = trial_goal, data_dir = data_dir, dual_policy=True)
            
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
    curiosity_based_train() 

            


