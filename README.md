# undergrad_paper
hanseul's undergrad research curiosity based RL for dynamic reward environment 
(Read the full thesis with the undergraduate_thesis.pdf file!)

## Introduction

This paper is my undergraduate thesis about comparing artificial agent(Deep Q Reinforcement Learning agent) and natural agent(Rhesus monkey) in 3D navigational task.
This paper shows novel curiosity maximum entropy reinforcement learning model which mixes goal-based external reward function and curiosity-based internal reward function in non-linear way. The model structure can be seen from figure below.

![image](https://github.com/k1seul/undergrad_paper/assets/117340073/347b13db-d070-4b59-bc6a-b37dfcf5da37)

## Experiment Design(Task Design) 

Rhesus monkeys were trained to solve 3D navigation task with multiple sub-reward and daily changing jackpot reward. This same environment was translated to 2d pygame for training reinforcement learning(RL) agents.
RL models which were used in this experiment are followings: Deep-Q-Network(DQN)(with or without batch learning), model-based RL, and curiosity-based RL which is explained in the introduction.


![image](https://github.com/k1seul/undergrad_paper/assets/117340073/2d68ff2d-711a-4a54-85cd-427c243d2a7b)
