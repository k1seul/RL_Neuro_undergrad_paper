from simple_dqn_train import simple_dqn_train
from model_based_train import model_based_train
from single_dqn_train import single_dqn_train
from rand_train import rand_dqn_train
from explore_agent_train import explore_dqn_train
import gc
import torch
from curiosity_based_train import curiosity_based_train


for i in range(8):
 #   simple_dqn_train(rand_seed=i, fixed_trial=i)
    simple_dqn_train(rand_seed=i, fixed_trial=i, bool_PER=True)
    torch.cuda.empty_cache()
    gc.collect()

"""

for i in range(10, 15):
    rand_dqn_train(rand_seed=i)
"""




for i in range(10, 15):
  #  simple_dqn_train(rand_seed=i)
   # torch.cuda.empty_cache()
   # gc.collect()
    simple_dqn_train(rand_seed=i,  bool_PER=True)
    torch.cuda.empty_cache()
    gc.collect()

"""

simulation_num = [2, 5, 10, 15, 20, 30]
simulation_max_episode = [30, 12, 6, 4, 3, 2]

for i in range(6):
    for l in range(2):
        model_based_train(rand_seed=l, simulation_num=simulation_num[i], simulation_max_episode=simulation_max_episode[i])
        torch.cuda.empty_cache()
        gc.collect()



for i in range(8):
    rand_dqn_train(rand_seed=i, fixed_trial=i)
    gc.collect()
sigmoid_weights = [2,3,4,5,10,20,30]
for sigmoid_weight in sigmoid_weights:
    explore_dqn_train(rand_seed=sigmoid_weight, sigmoid_weight=sigmoid_weight, sigmoid=True)
    gc.collect()




for i in range(8):
    simple_dqn_train(rand_seed=i, fixed_trial=i, bool_PER=True)
    torch.cuda.empty_cache()
    gc.collect()


for i in range(10, 20):
    simple_dqn_train(rand_seed=i, bool_PER=True)
    torch.cuda.empty_cache()
    gc.collect()

for i in range(2):
    curiosity_based_train(rand_seed=i, random_explore= False)
    torch.cuda.empty_cache()
    gc.collect()


for i in range(30,32):
    curiosity_based_train(rand_seed=i, random_explore= True)
    torch.cuda.empty_cache()
    gc.collect()

"""