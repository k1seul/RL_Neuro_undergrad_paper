from .network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 
import pickle

class VanilaDqnAgent():
    """ implementation of simple dqn agent with experience replay and per"""
    def __init__(self, state_size, action_size, hyperparameters, policy_network= "Q_network", 
                 bool_PER = False, seed_value = 0
                 ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running the calculation on " + str(self.device))
        self.weight_data_dir = None

        ## Agent action mask on or off 
        self.action_mask_bool = False


        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hyperparameters["hidden_size"]
        self.learning_rate = hyperparameters["learning_rate"]

        ## PER hyperparameters (change it later)
        self.TD_epsilon = hyperparameters["PER_TD_epsilon"]
        self.alpha = hyperparameters["PER_alpha"]


        self.batch_size = hyperparameters["batch_size"]
        self.memory_size = hyperparameters["memory_size"] 
        self.gamma = hyperparameters["gamma"] 
        self.epsilon = hyperparameters["epsilon"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.epsilon_decay_rate = hyperparameters["epsilon_decay_rate"]
        self.bool_PER = bool_PER 


        ## expected reward mutiple 

        self.expected_reward_alpha = hyperparameters["expected_reward_alpha"] 


        if self.bool_PER:
            """ if PER is used td error of each action - estimated q value is saved to the memory """
            self.td_error_memory = deque(maxlen = self.memory_size)

        # exeprience memory for the batch learning (batch is randomly sampled from the memory)
        self.experience_memory = deque(maxlen=self.memory_size)


        if policy_network == 'Q_network':
            print("policy network is currently q network")
            self.q_network = QNetwork(state_size, action_size, self.hidden_size, seed=seed_value).to(self.device)
        else:
            raise Exception('Error!!!! network not defined')
        

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def random_seed(self, seed_value):
        """set the random seed for the torch and numpy""" 
        np.random.seed(seed_value)
        random.seed(seed_value)

    def act(self, state, info = None, max_option=False, return_q_values = False):
        """max_option is for the testing purpose(also for plotting), 
        if max_option is True, then the agent will always choose the action with the highest q value 
        added info as action mask"""


        if not(self.action_mask_bool):
            info = np.ones([4]) 
        if np.random.rand() <= self.epsilon and not(max_option):
            if not(self.action_mask_bool):
                return np.random.randint(self.action_size) 
            else:
                return np.random.choice(np.where(info==1)[0])
        else:
            state_tensor = torch.Tensor(state).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                q_values = q_values.cpu().numpy() * info ## info in the action mask 
                max_action = np.argmax(q_values) 
                max_q = q_values[max_action]
          
                return max_action
    
    def get_expected_reward(self, state, action): 
        state_tensor = torch.Tensor(state).to(self.device) 
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy() 
            q_value_of_action = q_values[action]
            return q_value_of_action * self.expected_reward_alpha
                
    
    
    def calculate_td_error(self, state, action, reward, next_state, done):
        """ for Priortized experience replay implementation """

        state = torch.Tensor(state).to(self.device)
        next_state = torch.Tensor(next_state).to(self.device)
        current_q_values = self.q_network(state) 
        next_q_values = self.q_network(next_state) 
        current_q = current_q_values[action]
        max_next_q = np.argmax((next_q_values).cpu().detach().numpy())
        max_next_q = next_q_values[max_next_q]
        td_error = abs(reward + self.gamma * max_next_q - current_q)
        priority = (td_error + self.TD_epsilon)*self.alpha
        priority = priority.cpu().detach().numpy() 

        self.td_error_memory.append(priority)
    
    def remember(self, state, action, reward, next_state, done):
        self.experience_memory.append((state, action, reward, next_state, done))
        if self.bool_PER:
            self.calculate_td_error(state, action, reward, next_state, done)


    def sample_priortised_experience_replay(self):
        """sampling from memory using td error as priortised experience replay weights
        output the p values of len(memory) for use as sampling weights"""
        memory = self.td_error_memory

        td_error_p = memory / (sum(memory))

        minibatch_idx = np.random.choice(list(range(len(self.experience_memory))), self.batch_size, p=td_error_p)
        minibatch = [self.experience_memory[i] for i in minibatch_idx]
        return minibatch
        

    def replay(self, TD_sample = False): 
        if len(self.experience_memory) < self.batch_size:
            return 
        elif not(TD_sample): 
            minibatch = random.sample(self.experience_memory, self.batch_size)
        else: 
            minibatch = self.sample_priortised_experience_replay()

        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        ## greedly optimized with TD error 
        q_values = self.q_network(states)
        next_q_values = self.q_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1))
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def single_replay(self):
        """ replay of single resent experience"""
        experience = self.experience_memory[-1]
        state = torch.FloatTensor(np.array([experience[0]])).to(self.device)
        action = torch.LongTensor([experience[1]]).to(self.device)
        reward = torch.FloatTensor([experience[2]]).to(self.device)
        next_state = torch.FloatTensor(np.array([experience[3]])).to(self.device)
        done = torch.FloatTensor([experience[4]]).to(self.device)

        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        target_q_values = reward + (1 - done) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, action.unsqueeze(1))
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay_rate  * self.epsilon)

    def save_network_weight(self, trial_num=0):
        if not(trial_num % 10 == 0 ):
            return 
        
        file_name = self.weight_data_dir + f"network_{str(trial_num)}.pkl"

        weights = []

        for param in self.q_network.parameters():
            if param.requires_grad:
                weights.append(param.data.cpu().view(-1).numpy())

        # Convert the list of weights into a numpy array
        weights_array = np.concatenate(weights)


        with open(file_name, 'wb') as f:
            pickle.dump(weights_array, f)
    
