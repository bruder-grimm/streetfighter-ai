import os
import random
from random import randint
from random import seed

import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from pympler.tracker import SummaryTracker

tracker = SummaryTracker()
csv_logger = CSVLogger('training.log', append=True, separator='|')

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


class Agent:
    def __init__(self, state_size, action_size, gamma, epsilon, learning_rate, log_dir, batch_size):
        self.replay_buffer = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        self.replay_buffer_size = 0

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.exploration_min = 0.01
        self.exploration_decay = 0.995

        self.max_experiences = 10000

        self.filepath_best = "model_weights_{}_{}_{}_best.hdf5".format(str(gamma), str(epsilon), str(learning_rate))
        self.filepath_last = "model_weights_{}_{}_{}_last.hdf5".format(str(gamma), str(epsilon), str(learning_rate))

        self.tb_call_back = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
        self.checkpoint = ModelCheckpoint(
            self.filepath_best,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            save_freq=10,
            mode='min'
        )
        self.reduce_lr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=10,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        )
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape=self.state_size))
        model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='RandomNormal'))

        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        # this lets us pick up training of the last model with the best loss
        if os.path.isfile(self.filepath_best):
            print("Loaded model weights from {}", self.filepath_best)
            model.load_weights(self.filepath_best)
        else:
            print("Initialized empty model")
        return model

    # save the model with suffix "_last"
    def save_model(self):
        self.model.save(self.filepath_last)

    # here, we will both explore random actions for learning, how often this happens can be tweaked with epsilon,
    # and execute some predictions from our learned state
    def get_action_for(self, state):
        if np.random.rand() <= self.epsilon:  # explore
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)  # update
        return np.argmax(act_values[0])

    def add_to_experience(self, state, action, reward, next_state, done):
        if isinstance(action, np.int64):
            action = action.astype(int)

        if len(self.replay_buffer['state']) >= self.max_experiences:
            for key in self.replay_buffer.keys():
                self.replay_buffer[key].pop(0)

        #  this is the only way to free memory because python memory management sucksssss
        if self.replay_buffer_size > self.max_experiences * 2:
            self._clear_replay_buffer()

        experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
        for key, value in experience.items():
            self.replay_buffer[key].append(value)

    def _clear_replay_buffer(self):
        print("Clearing replay buffer...")
        copy = self.replay_buffer.copy()
        del self.replay_buffer
        self.replay_buffer = copy
        gc.collect()
        tracker.print_diff()
        print("Done!")

    # update the q values in the table
    # We predict a q value, multiply it with the discount factor depending on how much weight we want to give our
    # predictions. Now, we want to minimize the MSE between our predicted q and the actual q, which is the
    # target (also defined as such in the method), this is all handled by model.fit
    # On terminal state, we update with the reward
    # will save the model with the prefix "_best" if loss has improved
    def train_on_experience(self):
        collected_experience = len(self.replay_buffer['state'])

        if collected_experience < self.batch_size:
            return

        seed(1)

        for _ in range(self.batch_size):
            sample = randint(0, collected_experience - 1)

            state = self.replay_buffer['state'][sample],
            action = self.replay_buffer['action'][sample],
            reward = self.replay_buffer['reward'][sample],
            next_state = self.replay_buffer['next_state'][sample],
            done = self.replay_buffer['done'][sample]

            q = reward  # terminal state the value gets updated with the reward
            if not done:
                q = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # we're having some type issues and no one knows why
            if isinstance(q, np.ndarray):
                q = q.item(0)
            elif isinstance(q, tuple):
                q = q[0]

            predicted_q = self.model.predict(state)
            predicted_q[0][action] = q  # update the q value for the executed action in the predicted

            self.model.fit(
                state,          # our q
                predicted_q,    # our q'
                verbose=0,
                callbacks=[self.checkpoint, self.tb_call_back, self.reduce_lr, MyCustomCallback(), csv_logger]
            )

        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay
