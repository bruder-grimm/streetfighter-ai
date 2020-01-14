import os
import random
from collections import deque

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

csv_logger = CSVLogger('training.log', append=True, separator='|')


class Agent:
    def __init__(self, state_size, action_size, gamma, epsilon, learning_rate, log_dir, batch_size):
        self.replay_buffer = deque(maxlen=2000)

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.exploration_min = 0.01
        self.exploration_decay = 0.995

        self.filepath_best = "model_weights_{}_{}_{}_best.hdf5".format(str(gamma), str(epsilon), str(learning_rate))
        self.filepath_last = "model_weights_{}_{}_{}_last.hdf5".format(str(gamma), str(epsilon), str(learning_rate))

        self.tb_call_back = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
        self.checkpoint = ModelCheckpoint(
            self.filepath_best,
            monitor='loss',
            verbose=1,
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
            model.load_weights(self.filepath_best)
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
        self.replay_buffer.append((state, action, reward, next_state, done))

    # update the q values in the table
    # We predict a q value, multiply it with the discount factor depending on how much weight we want to give our
    # predictions. Now, we want to minimize the MSE between our predicted q and the actual q, which is the
    # target (also defined as such in the method), this is all handled by model.fit
    # On terminal state, we update with the reward
    # will save the model with the prefix "_best" if loss has improved
    def train_on_experience(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        sample_batch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in sample_batch:
            q = reward  # terminal state the value gets updated with the reward
            if not done:
                q = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            predicted_q = self.model.predict(state)
            predicted_q[0][action] = q  # update the q value for the executed action in the predicted

            self.model.fit(
                state,          # our q
                predicted_q,    # our q'
                epochs=1,
                verbose=1,
                callbacks=[self.checkpoint, self.tb_call_back, self.reduce_lr, csv_logger]
            )

        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay
