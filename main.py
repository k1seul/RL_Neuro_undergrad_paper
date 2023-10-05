from simple_dqn_train import simple_dqn_train
from model_based_train import model_based_train

for i in range(8):
    simple_dqn_train(rand_seed=i, fixed_trial=i)


for i in range(1):
    simple_dqn_train(rand_seed=i)


simulation_num = [2, 5, 10, 15, 20, 30]
simulation_max_episode = [30, 12, 6, 4, 3, 2]

for i in range(6):
    model_based_train(rand_seed=i, simulation_num=simulation_num[i], simulation_max_episode=simulation_max_episode[i])

