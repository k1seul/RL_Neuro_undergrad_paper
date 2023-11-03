from simple_dqn_train import simple_dqn_train
from model_based_train import model_based_train
from single_dqn_train import single_dqn_train
from rand_train import rand_dqn_train
from explore_agent_train import explore_dqn_train
import gc


for i in range(8):
    single_dqn_train(rand_seed=i, fixed_trial=i)
    gc.collect()

for i in range(10, 20):
    single_dqn_train(rand_seed=i)
    gc.collect()

simulation_num = [2, 5, 10, 15, 20, 30]
simulation_max_episode = [30, 12, 6, 4, 3, 2]

for i in range(6):
    for l in range(1):
        model_based_train(rand_seed=l, simulation_num=simulation_num[i], simulation_max_episode=simulation_max_episode[i])
        gc.collect()


"""
for i in range(8):
    rand_dqn_train(rand_seed=i, fixed_trial=i)
    gc.collect()
sigmoid_weights = [2,3,4,5,10,20,30]
for sigmoid_weight in sigmoid_weights:
    explore_dqn_train(rand_seed=sigmoid_weight, sigmoid_weight=sigmoid_weight, sigmoid=True)
    gc.collect()





for i in range(8):
    simple_dqn_train(rand_seed=i, fixed_trial=i, bool_PER=True)
    gc.collect()


for i in range(10, 20):
    simple_dqn_train(rand_seed=i, bool_PER=True)
    gc.collect()
"""