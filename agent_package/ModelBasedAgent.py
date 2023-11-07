from .network_module import * 
import torch
import torch.nn as nn 
import torch.optim as optim 
from collections import deque 
import random 
import numpy as np 
import pickle


class cross_ent_loss(nn.Module):
    """ for cross entropy update between exploit and explore"""
    def __init__(self):
        super(cross_ent_loss, self).__init__()
        self.eps = 1e-7
        
    def forward(self, exploit_q_values , explore_q_values):
        
        soft_max_exploit = torch.softmax(exploit_q_values, dim=0)
        soft_max_explore = torch.softmax(explore_q_values, dim=0)

        log_exploit_prob = torch.log(soft_max_exploit + self.eps)

        loss = -torch.mean(log_exploit_prob * soft_max_explore)
        ##print("advantage:", advantage, "prob:", soft_max_p_values, "'\nloss:", loss, "\nwtf:", wtf)

        return loss 



class ModelBasedAgent():
    """ simple DQN agent """
    def __init__(self, state_size, action_size, env_size, hyperparameters,
                 policy_network="Q_network",
                 explore_agent = False,
                 seed_value = 0): 
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("currently running on" + str(self.device))

        self.gpu_usage = True if torch.cuda.is_available() else False 


        self.action_mask_bool = False 
        self.weight_data_dir = None

        self.alpha = hyperparameters["PER_alpha"]
        self.TD_epsilon = hyperparameters["PER_TD_epsilon"] 
        self.state_size = state_size 
        self.action_size = action_size
        self.env_size = env_size ## size of the grid_world env for model implementation 

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


        
        self.experience_memory = deque(maxlen=self.memory_size)


        self.model_memory = deque(maxlen=self.memory_size)
        if explore_agent:
            self.explore_memory = deque(maxlen=self.memory_size)

        # td_error memory for calculating experience repaly weights 
        self.td_error_memory = deque(maxlen=self.memory_size)

        ## initializing policy network of choosing

        if policy_network == 'Q_network':
            print("policy network is currently q network")
            self.q_network = QNetwork(self.state_size, action_size, self.hidden_size, seed=seed_value).to(self.device)
        else:
            raise Exception('Error!!!! network not defined')
        


        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if explore_agent:
            self.__init_exploration_policy(seed_value=seed_value) 

    def random_seed(self, seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)


    def __init_exploration_policy(self, seed_value = 0): 

        self.explore_q_network = QNetwork(self.state_size, self.action_size, self.hidden_size, seed=seed_value).to(self.device)
        self.explore_optimizer = optim.Adam(self.explore_q_network.parameters(), lr=self.learning_rate)
        self.explore_criterion = nn.MSELoss() 

        self.max_ent_optim = optim.Adam(self.q_network.parameters(), lr= 0.0001)
        self.max_ent_loss = cross_ent_loss()


    def resent_memory_to_counter_matrix(self, memory_size = 100):

        counter_matrix = np.zeros([self.env_size, self.env_size])
        
        resent_memory_batch = list(self.experience_memory)[-2-memory_size: -1]
        resent_memory = [t[0] for t in resent_memory_batch]
        
        if len(resent_memory) == 0:
            return counter_matrix
        
        for state in resent_memory:
            counter_matrix[state[0]][state[1]] += 1 


        return counter_matrix


    def act(self, state, info=None, max_option=False):


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
                q_values = q_values.detach().cpu().numpy() * info 
                return np.argmax(q_values)
            
    def explore_act(self, state, info,  max_option=False):

        if np.random.rand() <= self.epsilon and not(max_option) :
            return np.random.randint(self.action_size)
        else:
            state_tensor = torch.Tensor(state).to(self.device)

            with torch.no_grad():
                q_values = self.explore_q_network(state_tensor)
                q_values = q_values.detach().cpu().numpy() * info 
                return np.argmax(q_values) 
        
            


    def remember(self, state, action, reward, next_state, done, model=False, explore=False):
        """ three memory for the agnet
            1. experience memory for updatinf using real experience (model=False, explore=False)
            2. model memory for updating using model experience (model=True, explore=False)
            3. explore memory for updating using explore experience (model=False, explore=True)"""
        if model:
            self.model_memory.append((state, action, reward, next_state, done))
        if explore:
            self.explore_memory.append((state, action, reward, next_state, done))            
        if not(model or explore): 
            self.experience_memory.append((state, action, reward, next_state, done))
            

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
        
    

    def sample_priortised_experience_replay(self, memory):
        """sampling from memory using td error as priortised experience replay weights
        output the p values of len(memory) for use as sampling weights"""

        td_error_p = memory / (sum(memory))

        minibatch_idx = np.random.choice(list(range(len(self.experience_memory))), self.batch_size, p=td_error_p)
        minibatch = [self.experience_memory[i] for i in minibatch_idx]
        return minibatch


    def replay(self, TD_sample = False, model=False): 
        if len(self.experience_memory) < self.batch_size:
            return 
        
        if len(self.model_memory) < self.batch_size and model:
            return 
        
        if model:
            minibatch = random.sample(self.model_memory, self.batch_size)
        elif not(TD_sample) :
            minibatch = random.sample(self.experience_memory, self.batch_size) 
        elif TD_sample:
            minibatch = self.sample_priortised_experience_replay(self.td_error_memory) 
        
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
        ## cliped weight 
        ##clipper = WeightClipper()
        ##self.q_network.apply(clipper)

    def explore_replay(self, td_sample=False):

        if len(self.explore_memory) < self.batch_size:
            return 
        
        if td_sample:
            NotImplementedError() 
        else:
            minibatch = random.sample(self.explore_memory, self.batch_size) 

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)


        ## greedly optimized with TD error 
        q_values = self.explore_q_network(states)
        next_q_values = self.explore_q_network(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1))
        loss = self.explore_criterion(q_values, target_q_values.unsqueeze(1))
        self.explore_optimizer.zero_grad()
        loss.backward()
        self.explore_optimizer.step()
    
    def maxent_replay(self, state, td_sample=False):
        state = torch.Tensor(state).to(self.device)
        exploit_q_values = self.q_network(state)
        explore_q_values = self.explore_q_network(state)

        loss = self.max_ent_loss(exploit_q_values, explore_q_values) 
        self.max_ent_optim.zero_grad()
        loss.backward()
        self.max_ent_optim.step()


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
    