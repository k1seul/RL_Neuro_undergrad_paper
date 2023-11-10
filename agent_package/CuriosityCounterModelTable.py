import numpy as np
from collections import deque 
import random 
from PIL import Image
from matplotlib import pyplot as plt 
import math 
from .ModelBasedAgent import ModelBasedAgent 
import os 



class CuriosityCounterModelTable(): 
    def __init__(self, table_size=11, memory_size=5, data_dir='img/', img_save=False):
        
      self.env_size = table_size 
      self.model_map = np.zeros([table_size, table_size])
      self.play_map = self.model_map
      self.play_small_reward = [] 
      self.exploration_memory = deque(maxlen=1000) 

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
          if (mem==goal_location).all():
              return True
    
    def reset(self, only_goal=False): 
       """ reset for every episode(of real env) goal and small reward locations"""
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
       
      map_changed = False if self.world_knowledge_before == self.world_knowledge_percentage(node_num = True) else True
      self.world_knowledge_before = self.world_knowledge_percentage(node_num = True)

      if self.isin_mem(self.reward_location_memory, state) and not(done):
        self.reward_location_memory.pop() 
        self.reset(only_goal = True)
        
      self.model_map[state[0]][state[1]] = 1

      if done:
         self.reward_location_memory.append(np.array(state)) 
      elif reward > 0 and not(self.known_small_reward(state = state)):
         self.small_reward_location_memory.append(np.array(state)) 
      if map_changed:
         self.draw_map(i_episode) 


    def create_frame(self, cognitive_map, episdoe_num,t):
      """
      function saving gray scale imaging of cognitive grid map"""

      data_dir = self.data_dir + f"model_image/"
      if not(os.path.exists(data_dir)):
         os.makedirs(data_dir)

      plt.imshow(cognitive_map*32, cmap='gray')
      plt.savefig(data_dir + f"episode_{episdoe_num}_{t}.png") 



    def draw_map(self, episode_num): 
      if not(self.img_save):
         return 
      self.cognitive_map = np.copy(self.model_map) 
      if len(self.reward_location_memory) == 0 :
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
         reward = -0.1
         return next_state, reward, done

      

      if not(self.reward_location is None) and (next_state == self.reward_location).all():
         
         reward = self.reward_size 
         done = True
      elif self.check_small_reward(next_state):
         reward = self.small_reward_size
         
      

      return next_state, reward, done 
    
    def curiosity_reward_simulate(self, state, action): 
       
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
      if self.play_map[new_location[0]][new_location[1]] == 1:
         next_state = new_location
      else:
         next_state = state 
      
      
      if np.array_equal(state, next_state): 
         wall_hit = True
         reward = -0.1 
      else:
         wall_hit = False

       

   
      reward = reward + self.calculate_curiosity_reward(next_state) 

      return next_state, reward, done 

    
    def calculate_curiosity_reward(self, next_state):
       
       counter_matrix = self.resent_memory_to_counter_matrix() 
       if np.sum(counter_matrix) == 0:
          return 0
       next_state_percentage = counter_matrix[next_state[0]][next_state[1]] / np.max(counter_matrix)
       info_reward = - 8 * math.log(next_state_percentage + 0.1) -4.08660499
       return info_reward
    

    def resent_memory_to_counter_matrix(self, memory_size = 1000):
        counter_matrix = np.zeros([11, 11])
        resent_memory_batch = list(self.exploration_memory)[-2-memory_size: -1] 
        resent_memory = resent_memory_batch

        if len(resent_memory) == 0: 
            return counter_matrix
        
        for state in resent_memory:
            counter_matrix[state[0]][state[1]] += 1

        return counter_matrix 

       
    
    def curiosity_simulate(self, agent=ModelBasedAgent, state=np.zeros(2), reset=True, exploit_update=False, maxent_update=True,
                           trial_num=None,data_saver=None, random_explore = False): 


      state_start = np.copy(state) 
      ## if you want to use the same memory for all simulations, uncomment the following line 
      ## (without memory deletion for every simulation)
      
      

      for epi_repeat in range(self.simulation_num):
         state = np.copy(state_start)
         
         
         episode_num = 0 

         if reset:
            goal_created = self.reset(only_goal=True)

         done = False 

         while not(done): 
            self.exploration_memory.append(state) 
            action = agent.explore_act(state= state, max_option=False, random_explore = random_explore) 
            next_state, reward, done = self.curiosity_reward_simulate(state, action)

            if episode_num >= self.simulation_max_episode:
               done = True 

            agent.remember(state, action, reward, next_state, done, model=False, explore=True)
            data_saver.record_visited_count(state=next_state, trial_num=trial_num, model=True, curiosity=True)
            agent.explore_replay() 
            state = np.copy(next_state)
            if exploit_update:
               if maxent_update:
                  agent.maxent_replay(next_state)
               else: 
                 agent.remember(state, action, reward, next_state, done, model=True, explore=False)
                 agent.replay(model=True)
               
            episode_num += 1 
            ##if episode_num % 15 == 0:
            ##   print(reward)

    
    def model_simulate(self, agent=ModelBasedAgent, state=np.zeros(2), reset=True, trial_num=None,data_saver=None, random = False):
        episode_num = 0
        if not(self.known_reward()):
            return

        start_state = np.copy(state)

        epsilon_save = agent.epsilon
        if random:
            agent.epsilon = 1

        for epi_repeat in range(self.simulation_num):
            state = np.copy(start_state)
            if reset:
                goal_created = self.reset(only_goal=True)

            done = False

            while not (done):
                action = agent.act(state)
                next_state, reward, done = self.simulate_map(state, action)

                if not(data_saver is None):
                    data_saver.record_visited_count(state=state, trial_num=trial_num, model=True)

                if episode_num >= self.simulation_max_episode:
                    done = True

                agent.remember(state, action, reward, next_state, done, model=True)
                agent.replay(model=True)
                episode_num += 1
                state = np.copy(next_state)

                
        if random:
            agent.epsilon = epsilon_save

        

    
          
      

  



       
       

       

        