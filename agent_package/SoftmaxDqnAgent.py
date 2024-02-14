import sys, os 
from .network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 

class SoftmaxDqnAgent():
    """softmax agent with experience replay and per"""
    def __init__(self, state_size, action_size, hidden_size, learning_rate,
                 memory_size, batch_size, gamma, policy_network= "Q_network", 
                 bool_PER = False 
                 ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running the calculation on " + str(self.device)) 


        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        ## PER hyperparameters (change it later)
        self.TD_epsilon = 0.0001
        self.alpha = 0.6 

        self.batch_size = batch_size 
        self.memory_size = memory_size 
        self.gamma = gamma 
        self.bool_PER = bool_PER 
        if self.bool_PER:
            self.td_error_memory = deque(maxlen = self.memory_size)

        # exeprience memory for the batch learning (batch is randomly sampled from the memory)
        self.experience_memory = deque(maxlen=self.memory_size)


        if policy_network == 'Q_network':
            print("policy network is currently q network")
            self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        elif policy_network == 'LSTM_Q':
            print("policy_network is currently LSTM network")
            self.q_network = LSTM_Q(state_size, action_size, hidden_size).to(self.device)
        else:
            raise Exception('Error!!!! network not defined')
        

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def random_seed(self, seed_value):
        """set the random seed for the torch and numpy"""
        np.random.seed(seed_value)
        random.seed(seed_value)



    def act(self, state, max_option=False): 

        state_tensor = torch.Tensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            if max_option:
                return np.argmax(q_values.cpu().numpy())
            else: 
                q_values = q_values - torch.mean(q_values)
                q_values = q_values / torch.std(q_values)
                p_values = torch.softmax(q_values, dim = 0).cpu().detach().numpy()
                return np.random.choice(list(range(self.action_size)), p = p_values)
    
    def calculate_td_error(self, state, action, reward, next_state, done):

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
        

    def replay(self, uniformed_sample = True, TD_sample = False): 
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
    