import os
import random
from collections import deque

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, state_size, action_size):
        self.weight_backup = "streetfighter_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 1e-2
        self.gamma = 0.99
        self.epsilon = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape=self.state_size))
        model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='RandomNormal'))

        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.epsilon = self.exploration_min
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def get_next_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):

        tb_call_back = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=False)
        csv_logger = CSVLogger('training.log', append=True, separator='|')

        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            q = reward  # last iteration
            if not done:
                q = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            predicted_q = self.model.predict(state)
            predicted_q[0][action] = q

            self.model.fit(state, predicted_q, epochs=1, verbose=1, callbacks=[tb_call_back, csv_logger])

        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay
