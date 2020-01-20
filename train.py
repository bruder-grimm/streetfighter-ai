import argparse
import os

import numpy as np
import retro
import tensorflow as tf
from Agent import Agent
from Actions import action_to_array


class Trainer:
    def __init__(self):
        self.log_dir = os.path.join(os.path.curdir, 'Graph')
        self.base_health = 176  # the base health for streetfighter in ram

        self.env = retro.make(
            'StreetFighterIISpecialChampionEdition-Genesis',
            scenario='scenario',
            obs_type=retro.Observations.RAM
        )

        self.episodes = 10000  # how many matches to play

        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.agent = Agent(
            self.input_size,
            self.output_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            log_dir='Graph',
            batch_size=32
        )

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def run(self, headless):
        try:
            for episode_index in range(self.episodes):
                done = False

                # this is the base observation
                state = self.env.reset()
                state = np.reshape(state, [1, self.input_size])

                # we want to observe health change from last to this state for reward
                last_enemy_health = self.base_health
                last_own_health = self.base_health

                # this is for logging
                all_rewards = np.empty(self.episodes)

                while not done:
                    action = self.agent.get_action_for(state)

                    # apply the predicted action to the sate and receive next_state
                    next_state, _, done, ram = self.env.step(action_to_array(action, self.output_size))

                    enemy_health = ram['enemy_health']
                    own_health = ram['health']

                    reward = 0

                    if enemy_health != last_enemy_health or own_health != last_own_health:
                        if enemy_health != self.base_health or own_health != self.base_health:

                            damage_reward = (last_enemy_health - enemy_health) / 5
                            avoidance_reward = (own_health - last_own_health) / 100

                            # our reward is defined by 'damage I inflict - damage I receive'
                            reward = damage_reward + avoidance_reward

                            if enemy_health == -1:
                                reward = reward * 2  # double reward for ko
                            elif own_health == -1:
                                reward = reward * 2  # ... in both ways

                            last_enemy_health = enemy_health
                            last_own_health = own_health

                    if not headless:
                        self.env.render()

                    # add the state and action to the memory to train on once the episode is done
                    next_state = np.reshape(next_state, [1, self.input_size])
                    self.agent.add_to_experience(next_state, action, reward, next_state, done)

                    # for tensorboard
                    all_rewards[episode_index] += reward
                    avg_rewards = all_rewards[max(0, episode_index - 100):(episode_index + 1)].mean()

                    with self.summary_writer.as_default():
                        tf.summary.scalar('episode reward', reward, step=episode_index)
                        tf.summary.scalar('running avg reward(100)', avg_rewards, step=episode_index)

                    state = next_state

                print("Episode {}# Reward: {}".format(episode_index, np.sum(all_rewards)))
                self.agent.train_on_experience() # before the next run we fit our model
        finally:
            # save the model on quitting training
            self.agent.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    args = parser.parse_args()

    streetfighter_training = Trainer()
    streetfighter_training.run(args.headless)
