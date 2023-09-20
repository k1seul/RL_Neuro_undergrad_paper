from MonkeyMazeEnv import MonkeyMazeEnv 
from MonkeyPath import MonkeyPath
import pygame 
import pickle
import numpy as np 
### For Playing the env with monkey!!! select the monkey name "p" or "s" and play within the env with them!
terminated = False
truncated = False 
env = MonkeyMazeEnv(render_mode="human", no_reward=False)
env.monkey_plot = True 
monkey_path = MonkeyPath(monkey_name="p")
trial_num = 10 

trial_start = monkey_path.get_start_position(trial_num)
trial_goal = monkey_path.get_goal_position(trial_num)
monkey_seq = monkey_path.get_trial(trial_num)
monkey_seq_max = len(monkey_seq)



trajectories = [] 
state, info = env.reset(reward = trial_goal, start = trial_start)
env._monkey_location = monkey_seq[0] 
terminated = False
truncated = False 
trajectories.append(state) 

pygame.event.clear()
sum_reward = 0
trial_len = 0 

while not(terminated or truncated):
    

    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_UP:
                action = 3
            elif ev.key == pygame.K_LEFT:
                action = 2
            elif ev.key == pygame.K_RIGHT:
                action = 0
            elif ev.key == pygame.K_DOWN:
                action = 1

            obs, reward, done, truncated, info = env.step(action)

            print(f"possible action: {np.where(info == 1)[0]}")
            print(action) 
            trial_len += 1 
            env._monkey_location = monkey_seq[min(trial_len, monkey_seq_max - 1)]
            print(f"obs: {obs}, reward:{reward}, info: {info}")
            sum_reward = sum_reward + reward

            trajectories.append(obs) 

    
            if done or truncated:
                print("Game over! Final score: {}".format(sum_reward))
                terminated = done 
                with open("example_trajectories", 'wb') as fp:
                    pickle.dump(trajectories, fp)
                    print("trajectories saved!")

                break 

while trial_len < monkey_seq_max - 1:
        trial_len += 1 
        env._monkey_location = monkey_seq[trial_len]
        env._render_frame() 



