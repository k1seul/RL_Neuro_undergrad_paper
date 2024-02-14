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



def curiosity_based_train_explore_control_2(rand_seed = 0, maxent=False, simulation_num = 3, simulation_max_episode = 20, explore_prob=0.1):

    gif_plotting = False      
    monkey_name = "p" 
    bool_pre_train = True 
    bool_PER = False 
    env = MonkeyMazeEnv() 
    monkey_path = MonkeyPath(monkey_name=monkey_name)
    trial_length = monkey_path.trial_num
    game_name = "Curiosity_counter_model_explore_control_2" if not(maxent) else "Curiosity_maxent_model_explore_control"
    game_name = game_name + str(explore_prob)
    

    data_dir, data_saver, writer = directory_setting(game_name, monkey_name,
        task_info=f"simple {game_name} training with pretrain {bool_pre_train} and PER {bool_PER}",
        run_num = rand_seed)

    ## agent_setting 
    state_size = env.state_n
    action_size = env.action_n
    hidden_size = 512
    learning_rate = 0.001
    memory_size = 10000
    batch_size = 128
    gamma = 0.99

    agent = agent_package.CuriosityBasedAgent(state_size = state_size,
                                            action_size = action_size,
                                            hidden_size = hidden_size,
                                            learning_rate = learning_rate,
                                            memory_size = memory_size,
                                            batch_size = batch_size,
                                            gamma = gamma,
                                            explore_agent = True,
                                            seed_value=rand_seed) 
    
    model = agent_package.CuriosityCounterModelTable(data_dir = data_dir)
    model.change_simulation_num(simulation_num = simulation_num, simulation_max_episode = simulation_max_episode)

    agent.random_seed(rand_seed)
    model.random_seed(rand_seed) 
    np.random.seed(rand_seed)

    if bool_pre_train:
        pre_train(agent=agent, model=model, epsilon_skip=True) 

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

            reward_known = model.known_reward() 

            if not(reward_known) or np.random.rand() < explore_prob:
                model.curiosity_simulate(agent, state, exploit_update=True, maxent_update=maxent)
            else: 
                model.model_simulate(agent, state) 
                model.curiosity_simulate(agent, state)
   
        

            plotting_functions.plot_all_functions(agent = agent, model = model, i_episode = trial_num,
                                                  trial_t = trial_t, state = state,
                                                  reward_location = trial_goal, data_dir = data_dir, dual_policy=True)
        
        while trial_t < trial_path_max - 1:
            trial_t += 1 
            env._monkey_location = trial_monkey_path[trial_t]
            env._render_frame() 

        path_analytic_tool.get_all_path_analytic_out(agent = agent, agent_path = state_trajectories, agent_action_seq = action_trajectories,
                monkey_path=trial_monkey_path, writer=writer, trial_num = trial_num, total_reward = total_reward, 
                total_length = total_length, data_saver = data_saver) 

        print("Episode: {}, total_reward: {:.2f}, length: {}".format(trial_num, total_reward, total_length))
        if gif_plotting: 
            env.recrdr.save()
        env.close()

    writer.close() 
    data_saver.save_data() 


if __name__ == "__main__":
    curiosity_based_train_explore_control() 

            


