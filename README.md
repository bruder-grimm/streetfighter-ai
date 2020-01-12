# StreetFighterAI

> ## Implementation Hints (DO NOT PUBLISH)
>
> - [x] PyTroch or Tensorflow 2.0 as framework
> - [x] Git as version control system (a development history should be visible, so commit often and early)
> - [ ] Clear function heads
> - [ ] Extensive code documentation
> - [ ] Referencing foreign program code
> - [x] Keep dependencies low
> - [x] Final results should be easy to reproduce (requirement.txt, dockerfile, ...)
>
> 
>
> ## **Documentation Hints** (DO NOT PUBLISH)
>
> - [x] git READ.md
> - [x] Description of the program and its purpose
> - [x] Installation and usage guidelines (you are responsible for a flawless running of the program and reproducibility of your results)
> - [ ] Further documentation (could part of the READ.md or the jupyter notebook or be another document)





## Description

This Tensorflow Keras Model uses OpenAI's Gym Retro Eviroment to train an Agent to play the Sega Genesis game *StreetFighterII - Special Champion Edition* using Deep Q-Learning.



## Get Started

These instructions help you set up the virtual env, install the requirements and set up the Retro Enviroment.



1. `mkdir StreetFighterAI`
2. create virtual venv
   `python3 -m venv StreetFighterAI/`
3. `cd StreetFighterAI/`
4. clone the repo into the same folder that holds the venv 
   `git clone git@github.com:bruder-grimm/ml.git`
5. activate the venv`source bin/activate`
6. go into the repository
   `cd ml`
7. install the requirements and run insert the rom
   `sh run.sh`



## Start the training

After you set up the environment, you can get started with the training. 



1. `cd StreetFighterAI/` (unless you are already)
2. start venv (unless you have already)
   `source bin/activate`
3. start training
   `python3 main.py`
4. open tensorboard for monitoring
    `tensorboard --logdir Graph/`











