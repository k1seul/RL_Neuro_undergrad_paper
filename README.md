# undergrad_paper
hanseul's undergrad research curiosity based RL for dynamic reward environment 
(Read the full thesis with the undergraduate_thesis.pdf file!)

keywords: Reinforcement Learning, Dynamic Environment, Curiosity, Natural-Artificial intelligence similarity, Representational Similarity Analysis

## Introduction

This paper is my undergraduate thesis about comparing artificial agent(Deep Q Reinforcement Learning agent) and natural agent(Rhesus monkey) in 3D navigational task.
This paper shows novel curiosity maximum entropy reinforcement learning model which mixes goal-based external reward function and curiosity-based internal reward function in non-linear way. The model structure can be seen from figure below.



<p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/347b13db-d070-4b59-bc6a-b37dfcf5da37">
</p>


## Experiment Design(Task Design) 

Rhesus monkeys were trained to solve 3D navigation task with multiple sub-reward and daily changing jackpot reward(change of location). This same environment was translated to 2d pygame for training reinforcement learning(RL) agents.
RL models which were used in this experiment are followings: Deep-Q-Network(DQN)(with or without batch learning), model-based RL, and curiosity-based RL which is explained in the introduction.


<p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/2d68ff2d-711a-4a54-85cd-427c243d2a7b">
</p>



Comparsion between natural agents(Rhesus monkeys) and artificial agents(multiple RL models) were done in two main ways. One being behavioral similarity which compares directed(non-random) components and efficiency of both agents.

<p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/ca212a34-8cf5-47df-92a2-9bd07e88ca2f">
</p>


Other comparsion was done on similarity of dynamical changes of neural signal of orbitofrontal cortex(OFC), which is known to represent reward values, and network weights of multiple RL agents.
Since the dimensions of these two data are different, Principal Component Analysis was used to process these data into 10 dimenonal arrays. After this preprocessing cosine similarity was calculated to make Representatioanl Dissimilarity Matrix(RDM) 
These RDM dis-similarity data were calculated using three metric of cosine similarity, kendall-tau α and pearson correlations

<p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/ed250415-0f98-4fef-b926-f25d9c7696d0">
</p>


## Interesting Results! 

### Behavioral Results 

My novel curiosity maxent model showed best performance along multiple RL agents. More specificially, it showed better adaptability to changing reward position despite of low epsilon(exploration) value. 

<p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/3284c922-ba94-4fd1-88f3-f89f0e559aa4">
</p>


  The curiosity maxent model also showed most similarity to the rhesus monkey behavior!
  (choice similarity comapres all of the choice sequence of monkey to that of RL agents. Max choice similarity compares only the action with maximum chosen number per state.)
  Figure below shows curiosity maxent agent having lowest shaanon value meaning that it shows most systematic behavior. This means that choice similarity of curiosity maxent model to the rhesus monkey is not result of random behavior.


  <p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/6488e0fe-928d-4ee7-bf5e-0e70b76f0aa8">
</p>

  ### Neurological Results! 

  Cool looking figure! 
  This is plot of Representational Dissimilarity Matrix of the rhesus monkey's OFC and RL agents' deep-network weights. Higher color means that the neurological state(or net network's state) is more different and darker color means that it's state are similiar. 

  <p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/063be460-d73b-4a85-8dd4-ad544e3c8485">
</p>

The RDM matrices of curiosity matent model showed most similairty to the rhesus monkey's OFC RDM on all three comparison metric(cosine similarity, kendall-tau α and pearson correlations)
Distribution plots from below shows similarity using that of randomly suffled RL network RDMS. This is done to calculate p-values of each similairty metric being higher than chance.

  <p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/a314ad0a-0c66-497f-988a-cc00c3351c38">
</p>

My final result to show is that OFC RDM matrix shows temporal clustering. This is based on question that i had about what component makes the OFC's neurological state similar? By clustering RDM hierarchal first, we can calculate total sum of difference with our difference function of choosing.
If this difference comes out to be low(p < 0.01 to randomly shuffle data), we can say that it has clustering effect to that component. The difference functions we used were difference of reward index and distance of reward location. As seen from figure below difference sum between reward distance was not statically significant but that of reward index was, thus temporal clustering is present if OFC RDM.


  <p align="center">
  <img width=700 src="https://github.com/k1seul/undergrad_paper/assets/117340073/1c21197c-0265-4448-981d-4bac871e83cf">
</p>




  


