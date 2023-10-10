from .network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 


class SuccessorRepresentAgent():
    def __init__(self, state_size, action_size, hidden_size, learning_rate,
                 memory_size, batch_size, gamma, state_length = 11
                 ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running the calculation on " + str(self.device))
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.state_length = state_length 

        ## PER hyperparameters (change it later)
        self.TD_epsilon = 0.0001
        self.alpha = 0.6 


        self.batch_size = batch_size 
        self.memory_size = memory_size 
        self.gamma = gamma 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay_rate = 0.99

        self.experience_memory = deque(maxlen=self.memory_size)
        
        self.reward_map = np.zeros((state_length, state_length))
        self.srq_network = SRQnetwork(state_size, action_size, hidden_size, state_length).to(self.device)
        self.criteria = nn.MSELoss()
        self.optimizer = optim.Adam(self.srq_network.parameters(), lr=self.learning_rate)

    def random_seed(self, seed_value):
        """set the random seed for the torch and numpy""" 
        np.random.seed(seed_value)
        random.seed(seed_value)

    def record_reward(self, next_state, reward):
        """record the reward in the reward map"""
        if reward < 0:
            return 
        self.reward_map[next_state[0], next_state[1]] = reward
    
    def calculate_expected_reward(self, tensor_out):
        """calculate the expected reward of each action"""
        tensor_out = tensor_out.cpu().detach().numpy()
        expected_reward = np.sum(self.reward_map * tensor_out, axis = (0,1)) 
        return expected_reward 
    
    def act(self, state, max_option=False):
        """choose the action based on the current state"""
        if np.random.rand() <= self.epsilon and not(max_option):
            return np.random.randint(self.action_size) 
        else: 
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            expected_reward = np.array([0.,0.,0.,0.])
            for action in range(4):

                tensor_out = self.srq_network(state, torch.Tensor([action]).unsqueeze(0).to(self.device))
                expected_reward[action] = self.calculate_expected_reward(tensor_out)

            if np.min(expected_reward) == 0 and np.max(expected_reward) == 0:
                action = np.random.randint(self.action_size)
            else:
                action = np.argmax(expected_reward)
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """remember the experience"""
        self.experience_memory.append((state, action, next_state, done))
        self.record_reward(next_state, reward)

    def replay(self):
        if len(self.experience_memory) < self.batch_size:
            return 
        else: 
            minibatch = random.sample(self.experience_memory, self.batch_size)

        

        for i in range(self.batch_size):
            memory = minibatch[i] 
            state = torch.from_numpy(memory[0]).float().unsqueeze(0).to(self.device)
            action = memory[1]
            next_state_numpy = memory[2]
            next_state = torch.from_numpy(memory[2]).float().unsqueeze(0).to(self.device)
            done = np.array(memory[3])

            current_state_representation = np.zeros((self.state_length, self.state_length))
            current_state_representation[next_state_numpy[0]][next_state_numpy[1]] = 1 
            current_state_representation = torch.FloatTensor(current_state_representation).to(self.device)

            current_sr = self.srq_network(state, torch.Tensor([action]).unsqueeze(0).to(self.device))
            max_action = self.act(next_state_numpy, max_option=True)
            next_sr = self.srq_network(next_state, torch.Tensor([max_actionßß]).unsqueeze(0).to(self.device))

            target_sr = current_state_representation + self.gamma * next_sr 
            
            loss = self.criteria(current_sr, target_sr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def decay_epsilon(self):
        """decay the epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon_decay_rate  * self.epsilon)





        
