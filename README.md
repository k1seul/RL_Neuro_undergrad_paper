# undergrad_paper
hanseul's undergrad research curiosity based RL for dynamic reward environment 
(Read the full thesis with the undergraduate_thesis.pdf file!)

keywords: Reinforcement Learning, Dynamic Environment, Curiosity, Natural-Artificial intelligence similarity, Representational Similarity Analysis

## Introduction

This paper is my undergraduate thesis about comparing artificial agent(Deep Q Reinforcement Learning agent) and natural agent(Rhesus monkey) in 3D navigational task.
This paper shows novel curiosity maximum entropy reinforcement learning model which mixes goal-based external reward function and curiosity-based internal reward function in non-linear way. The model structure can be seen from figure below.

![image](https://github.com/k1seul/undergrad_paper/assets/117340073/347b13db-d070-4b59-bc6a-b37dfcf5da37)

## Experiment Design(Task Design) 

Rhesus monkeys were trained to solve 3D navigation task with multiple sub-reward and daily changing jackpot reward(change of location). This same environment was translated to 2d pygame for training reinforcement learning(RL) agents.
RL models which were used in this experiment are followings: Deep-Q-Network(DQN)(with or without batch learning), model-based RL, and curiosity-based RL which is explained in the introduction.


![image](https://github.com/k1seul/undergrad_paper/assets/117340073/2d68ff2d-711a-4a54-85cd-427c243d2a7b)

Comparsion between natural agents(Rhesus monkeys) and artificial agents(multiple RL models) were done in two main ways. One being behavioral similarity which compares directed(non-random) components and efficiency of both agents.

<img width="572" alt="image" src="https://github.com/k1seul/undergrad_paper/assets/117340073/ca212a34-8cf5-47df-92a2-9bd07e88ca2f">

Other comparsion was done on similarity of dynamical changes of neural signal of orbitofrontal cortex(OFC), which is known to represent reward values, and network weights of multiple RL agents.
Since the dimensions of these two data are different, Principal Component Analysis was used to process these data into 10 dimenonal arrays. After this preprocessing cosine similarity was calculated to make Representatioanl Dissimilarity Matrix(RDM) 
These RDM dis-similarity data were calculated using three metric of cosine similarity, kendall-tau α and pearson correlations

<img width="750" alt="image" src="https://github.com/k1seul/undergrad_paper/assets/117340073/ed250415-0f98-4fef-b926-f25d9c7696d0">

## Interesting Results! 

My novel curiosity maxent model showed best performance along multiple RL agents. More specificially, it showed better adaptability to changing reward position despite of low epsilon(exploration) value. 

![image](https://github.com/k1seul/undergrad_paper/assets/117340073/3284c922-ba94-4fd1-88f3-f89f0e559aa4)

