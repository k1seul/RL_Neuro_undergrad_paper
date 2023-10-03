import numpy as np
from collections import deque
import random
from matplotlib import pyplot as plt
import math
from .ModelBasedAgent import ModelBasedAgent
import os


class VanillaModelTable:
    def __init__(self, table_size=11, memory_size=5, data_dir="img/", img_save=False):

        self.env_size = table_size
        self.model_map = np.zeros([table_size, table_size])
        self.play_map = self.model_map
        self.play_small_reward = []
        self.road_index = 1
        self.reward_location = None
        self.reward_location_memory = deque(maxlen=memory_size)
        self.small_reward_location_memory = deque(maxlen=20)
        self.reward_size = 8
        self.small_reward_size = 1
        self.memory_size = memory_size
        self.world_knowledge_before = 0
        self.known_node_num_mem = 0
        self.information_reward = 1
        self.t = 0

        self.simulation_num = 3
        self.simulation_max_episode = 20 

        self.data_dir = data_dir
        self.img_save = img_save

    def random_seed(self, seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)
    
    def change_simulation_num(self, simulation_num, simulation_max_episode):
         self.simulation_num = simulation_num
         self.simulation_max_episode = simulation_max_episode

    def world_knowledge_percentage(self, node_num=False, percentage=False):
        """
        number is sum of all the map
        shannon info is calculated using suprising amount of unknown of known information
        """
        if node_num:
            return np.sum(self.model_map)
        elif percentage: 
            return np.sum(self.model_map) / 121 
        else:
            return True if np.sum(self.model_map) / 121 > 0.1 else False
        
    def known_reward(self):
      known = False if len(self.reward_location_memory) == 0 else True
      return known 


    def known_small_reward(self, state):
        known = False
        if len(self.small_reward_location_memory) == 0:
            return False

        for i, small_reward_pos in enumerate(self.small_reward_location_memory):
            if (np.array(small_reward_pos) == state).all():
                return True

        return False

    def isin_mem(self, memory, goal_location):
        if len(memory) == 0:
            return False

        for mem in memory:
            if (mem == goal_location).all():
                return True

    def reset(self, only_goal=False):
        """reset for every episode(of real env) goal and small reward locations"""
        self.play_map = np.copy(self.model_map)
        goal_created = False

        if len(self.reward_location_memory) == 0:

            self.reward_location = None
        else:
            self.reward_location = random.sample(self.reward_location_memory, 1)[0]
            goal_created = True

        if only_goal:
            return goal_created

        self.play_small_reward = self.small_reward_location_memory
        return goal_created

    def record_map(self, state, reward, done, i_episode):

        map_changed = (
            False
            if self.world_knowledge_before
            == self.world_knowledge_percentage(node_num=True)
            else True
        )
        self.world_knowledge_before = self.world_knowledge_percentage(node_num=True)

        if self.isin_mem(self.reward_location_memory, state) and not (done):
            self.reward_location_memory.pop()
            self.reset(only_goal=True)

        self.model_map[state[0]][state[1]] = 1

        if done:
            self.reward_location_memory.append(np.array(state))
        elif reward > 0 and not (self.known_small_reward(state=state)):
            self.small_reward_location_memory.append(np.array(state))
        if map_changed:
            self.draw_map(i_episode)

    def create_frame(self, cognitive_map, episdoe_num, t):
        """
        function saving gray scale imaging of cognitive grid map"""

        data_dir = self.data_dir + f"model_image/"
        if not (os.path.exists(data_dir)):
            os.makedirs(data_dir)

        plt.imshow(cognitive_map * 32, cmap="gray")
        plt.savefig(data_dir + f"episode_{episdoe_num}_{t}.png")

    def draw_map(self, episode_num):
        if self.img_save == False:
            return 
        self.cognitive_map = np.copy(self.model_map)
        if len(self.reward_location_memory) == 0:
            pass
        else:
            reward_location = random.sample(self.reward_location_memory, 1)[0]

            self.cognitive_map[reward_location[0]][reward_location[1]] = 8

        if episode_num < 50:
            self.create_frame(self.cognitive_map, episode_num, self.t)
        elif self.t % 50 == 0:
            self.create_frame(self.cognitive_map, episode_num, self.t)

        self.t = self.t + 1

    def check_small_reward(self, state):
        if len(self.play_small_reward) == 0:
            return False
        for i, small_reward_pos in enumerate(self.play_small_reward):
            if (np.array(small_reward_pos) == state).all():
                del self.play_small_reward[i]
                return True

        return False

    def simulate_map(self, state, action):

        reward = 0
        done = False
        movement = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        movement_vector = movement[action]
        new_location = state + movement_vector
        new_location = np.clip(new_location, 0, self.env_size - 1)

        if self.model_map[new_location[0]][new_location[1]] == 1:
            next_state = new_location
        else:
            next_state = state

        if (
            not (self.reward_location is None)
            and (next_state == self.reward_location).all()
        ):

            reward = self.reward_size
            done = True
        elif self.check_small_reward(next_state):
            reward = self.small_reward_size
        elif (next_state == state).all():
            reward = -0.1


        return next_state, reward, done

    def model_simulate(self, agent=ModelBasedAgent, state=np.zeros(2), reset=True):
        episode_num = 0

        start_state = np.copy(state)
        for epi_repeat in range(self.simulation_num):
            state = np.copy(start_state)
            if reset:
                goal_created = self.reset(only_goal=True)

            done = False

            while not (done):
                action = agent.act(state)
                next_state, reward, done = self.simulate_map(state, action)

                if episode_num >= self.simulation_max_episode:
                    done = True

                agent.remember(state, action, reward, next_state, done, model=True)
                agent.replay(model=True)
                episode_num += 1
                state = np.copy(next_state)
