import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 


"""
This Modules deals with multiple network to be used in maze task.
main policy network is simple linear DQN and model_network does only one step prediction 
"""



# Q network(main network) model structure 
# Inputs: state Outputs: action(q_values for each action)
# composed of 5 linear layer with selected hidden_size 
# forward method used relu as activation function

Network_names = ["QNetwork", "ShallowQnetwork" ,"LSTM_Q", "Model_Network_valila", "Model_Network_cross"]

class QNetwork(nn.Module):
    ## input dimension of each state, action, hidden. if the Q_Network is passed though the output of model, model_output_n need to be determined 
    def __init__(self, state_size, action_size, hidden_size, seed=0):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed) 
        self.fc1 = nn.Linear(state_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_size)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.fc5.weight)

    
    ## all foward method has relu activation function 
    def forward(self, x):
       
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
     
        
        
        return x
    
class softmax_QNetwork(nn.Module):
    ## input dimension of each state, action, hidden. if the Q_Network is passed though the output of model, model_output_n need to be determined 
    def __init__(self, state_size, action_size, hidden_size):
        super(softmax_QNetwork, self).__init__()
        hidden_size = 32 
        self.fc1 = nn.Linear(state_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.min_threshold = 0.01
    
    ## all foward method has relu activation function 
    def forward(self, x):
       
        
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1) 
        return x
    

class ShallowQnetwork(nn.Module): 
    ## input dimension of each state, action, hidden. if the Q_Network is passed though the output of model, model_output_n need to be determined 
    ## Networksize is (state_n + model_output_n)*(hidden_size)^layers=5*action_size 
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    ## all foward method has relu activation function 
    def forward(self, x):
       
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
     
        
        
        return x
    
class SRQnetwork(nn.Module):
    """Q network for successor representation agent input size is state size , output is n*n
    successor representation matrix with channel number of action size"""
    def __init__(self, state_size, action_size, hidden_size, state_length, seed=0):

        super(SRQnetwork, self).__init__()
        torch.manual_seed(seed)
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size 
        self.state_length = state_length 

        self.fc1 = nn.Linear(state_size + 1,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size) 
        self.fc5 = nn.Linear(hidden_size, state_length*state_length)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.fc5.weight)
        


    def forward(self, state, action): 
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(self.state_length, self.state_length)

        return x


## LSTM module with built in memory function
## has linear layer to transform hidden dim to action dim at final output 
class LSTM_Q(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, num_layers=4, batch_first=True):
        super(LSTM_Q, self).__init__() 

        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.action_size = action_size 
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm_hidden = None 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden=None):
        ## initialize hidden state with zeros 
        ## for fixed h0, c0 for stable update 
        ## h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        ## c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, hidden = self.lstm(x, self.lstm_hidden)  

        ## flatten of lstm output for fc layer 

        out = out.contiguous().view(-1, self.hidden_size)

        ## fc layer pass 

        out = self.fc(out)

        if self.lstm_hidden == None:
            self.lstm_hidden = hidden 


        return out, hidden 
     


    

# Model network is network chich inputs state, action and guesses next_state and reward
# if done_guess=True, it also guesses if the agent is in terminal state or not

class Model_Network_valila(nn.Module):
    def __init__(self, state_dim, output_dim, hidden_dim):
        super(Model_Network_valila, self).__init__()
        self.state_dim = state_dim

        self.fc1 = nn.Linear(state_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, model_input):


        x = torch.relu(self.fc1(model_input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)

        
        next_state, reward = torch.split(x, [self.state_dim - 1 , 1], dim=-1)
        return next_state, reward
    
    ### this function calculates all possible outcome of network from given state and possible actions and concate them 
    def output_all_action(self, state, gpu_usage): 

            

        output_matrix = [] 

        for action in range(4):

            action = torch.tensor(action, dtype=torch.float32, device='cuda:0').unsqueeze(0) if gpu_usage else torch.tensor(action).unsqueeze(0)
            action = action.unsqueeze(0) 

            

            if self.done_guess:
                next_state, reward, done = self.forward(state, action)
                out_vector = torch.cat((next_state, reward, done), dim=-1)
            else: 
                next_state, reward = self.forward(state, action)
                out_vector = torch.cat((next_state, reward), dim=-1)

            output_matrix.append(out_vector.squeeze(0))

        out = torch.cat(output_matrix, dim=0).unsqueeze(0).detach() 

        return(out)


class WeightClipper(object):
    def __call__(self,module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0, 100)
            module.weight.data = w


    

        

    

