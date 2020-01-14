# StreetFighterAI


## Description

This Tensorflow Keras Model uses OpenAI's Gym Retro Eviroment to train an Agent via Deep Q Learning to play the Sega Genesis game *StreetFighter II - Special Champion Edition*.


## Setup

These instructions help you set up the virtual env, install the requirements and set up the Retro Enviroment.



1. `mkdir StreetFighterAI`
2. create virtual venv
   `python3 -m venv StreetFighterAI/`
3. `cd StreetFighterAI/`
4. clone the repo into the same folder that holds the venv 
   `git clone git@github.com:bruder-grimm/ml.git`
5. activate the venv with `source bin/activate`

## Start the training

After you set up the environment, you can get started with the training. 

1. `cd StreetFighterAI/` (unless you are already)
2. start venv (if you haven't already)
   `source bin/activate`
3. start training
   `./train.sh`
4. open tensorboard for monitoring
    `tensorboard --logdir Graph/`
5. go for lunch, then maybe dinner, and if you're as unlucky as we are even breakfast
    
This will generate two output models like `model_weights_0.99_1.0_0.001_best.hdf5` and `model_weights_0.99_1.0_0.001_last.hdf5`,
the latter of which just means that this is the last one saved, regardless of it's loss
(so best naturally is the one with the lowest loss).

## Let the model play

Once you've trained for a sufficient amount of episodes, try your model by running
`python play.py --model your_model.hdf5`











