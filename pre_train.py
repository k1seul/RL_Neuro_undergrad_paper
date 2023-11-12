import numpy as np
from torch.utils.tensorboard import SummaryWriter
from agent_package.CuriosityCounterModelTable import CuriosityCounterModelTable as Model 
from agent_package.ModelBasedAgent import ModelBasedAgent as Agent
from MonkeyMazeEnv import MonkeyMazeEnv


def pre_train(agent = Agent, model = Model, episode_num = 10, epsilon_skip=False): 
    max_episode = 100
    env = MonkeyMazeEnv(no_reward=True)
    env.max_episode_step = max_episode

    if model is None: 
        model_based = False 
    else: 
        model_based = True 

    for i_episode in range(-1, -episode_num-1, -1):
        state, info = env.reset() 
        done = False
        truncated = False 
        total_length = 1 
        total_reward = 0 
        if model_based: 
            model.reset() 
            model.t = 0
            model.draw_map(i_episode)
            model.record_map(state=state, reward=0, done=done, i_episode=i_episode)

        while not(done or truncated): 
            action = agent.act(state) 
            next_state, reward, done, truncated, info = env.step(action) 

            if model_based: 
                model.record_map(state=next_state, reward=reward, done=False, i_episode=i_episode) 

            if reward >= 0:
                reward = 0 

            total_reward += reward 
            total_length += 1

            agent.remember(state, action, reward, next_state, done)
            if model_based and model.world_knowledge_percentage():
                try:
                    model.curiosity_simulate(agent=agent, state=next_state, exploit_update = True, pre_train=True)
                   
                except:
                    model.model_simulate(agent=agent, state=next_state)
            agent.replay() 

            state = next_state
        
        if not(epsilon_skip):
            agent.decay_epsilon()
            print("Episode: {}, total_reward: {:.2f}, epsilon: {:.2f}, length: {}".format(i_episode, total_reward, agent.epsilon, total_length))
        else:
            print("Episode: {}, total_reward: {:.2f}, length: {}".format(i_episode, total_reward,  total_length))
    if not(epsilon_skip):
        agent.epsilon = 0.8