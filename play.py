import argparse
import os

import numpy as np
import retro
from Actions import action_to_array
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential


class Player:
    def __init__(self, filepath):
        self.env = retro.make(
            'StreetFighterIISpecialChampionEdition-Genesis',
            scenario='scenario',
            obs_type=retro.Observations.RAM
        )
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.model = Sequential()

        self.model.add(InputLayer(input_shape=self.input_size))
        self.model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        self.model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        self.model.add(Dense(self.output_size, activation='linear', kernel_initializer='RandomNormal'))

        self.model.compile(loss='mse')

        self.noop = np.zeros(self.output_size)
        self.frameskip_enabled = True
        self.frameskip = 7

        # this lets us pick up training of the last model with the best loss
        if os.path.isfile(filepath):
            self.model.load_weights(filepath)

    def run(self):
        state = self.env.reset()
        state = np.reshape(scale(state), [1, self.input_size])

        done = False

        while not done:
            act_values = self.model.predict(state)

            action = action_to_array(np.argmax(act_values[0]), self.output_size)
            print("Action: ", action)

            observation, _, done, _ = self.env.step(action)
            self.env.render()

            # skip n frames so that the observed s' is actually a consequence of s(a)
            if self.frameskip_enabled:
                for _ in range(self.frameskip):
                    if not done:
                        observation, _, done, ram = self.env.step(self.noop)

            state = np.reshape(scale(observation), [1, self.input_size])


def scale(x):
    return x / 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Name of the model in this directory')
    args = parser.parse_args()

    streetfighter_play = Player(args.model)
    streetfighter_play.run()
