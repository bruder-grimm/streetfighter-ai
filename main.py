import retro
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from DQLAgent import Agent

def action_to_array(action):

        # define some cool streetfighter actions here

        action_set = {
            0: np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
            1: np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
            2: np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]),
            3: np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
            4: np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]),
            5: np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
            6: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            7: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
            8: np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
            9: np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
            10: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            11: np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]),
        }

        return action_set[action]

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000
        self.env               = nv = retro.make(
        'StreetFighterIISpecialChampionEdition-Genesis',
        scenario='scenario',
        obs_type=retro.Observations.RAM
    )
        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0



                while not done:
                    self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action_to_array(action))
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()