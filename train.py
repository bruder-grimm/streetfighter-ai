import argparse
import os

import numpy as np
import retro
import tensorflow as tf
from Actions import action_to_array
from Agent import Agent

base_health = 176  # the base health for streetfighter in ram


class Trainer:
    def __init__(self):
        self.log_dir = os.path.join(os.path.curdir, 'Graph')

        self.env = retro.make(
            'StreetFighterIISpecialChampionEdition-Genesis',
            scenario='scenario',
            obs_type=retro.Observations.RAM
        )

        self.episodes = 10000  # how many matches to play

        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.noop = np.zeros(self.output_size)
        start = np.zeros(self.output_size)
        start[5] = 1
        self.start = start


        self.frameskip_enabled = False
        self.frameskip = 7

        self.agent = Agent(
            self.input_size,
            self.output_size,
            learning_rate=0.01,
            gamma=0.85,
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
                state = np.reshape(scale(state), [1, self.input_size])

                # we want to observe health change from last to this state for reward
                last_enemy_health = base_health
                last_own_health = base_health

                # this is for logging
                episode_reward = 0

                last_action = 0

                # this is for frameskip
                frame = -1

                while not done:
                    frame += 1

                    # implement frameskip because actions take some frames to hit in Streetfighter
                    if not frame % self.frameskip == 0 and self.frameskip_enabled:
                        next_state, _, done, ram = self.env.step(self.noop)

                        if not done:  # advance the state s_prime to be s_prime + frameskip
                            state = np.reshape(scale(next_state), [1, self.input_size])
                        else:  # premature end, we usually hope that the game doesn't end during skipped frames
                            print("Episode {}# done during skipped frame, rewards may be off".format(episode_index))
                            state = self.reshape_next_state_and_add_to_model_experience(
                                state,
                                last_action,
                                get_reward(ram['enemy_health'], last_enemy_health, ram['health'], last_own_health),
                                next_state,
                                done
                            )
                        continue

                    last_action = self.agent.get_action_for(state)

                    # apply the predicted action to the sate and receive next_state
                    next_state, _, done, ram = self.env.step(action_to_array(last_action, self.output_size))

                    enemy_health = ram['enemy_health']
                    own_health = ram['health']

                    reward = get_reward(enemy_health, last_enemy_health, own_health, last_own_health)

                    if own_health < 0 or enemy_health < 0:
                        print("Round over, got {} reward without KO".format(episode_reward))
                        while not done or ram['enemy_health'] == base_health and ram['own_health'] == base_health:
                            next_state, _, done, ram = self.env.step(self.start)

                        state = np.reshape(scale(next_state), [1, self.input_size])
                        last_enemy_health = base_health
                        last_own_health = base_health

                    else:
                        last_enemy_health = enemy_health
                        last_own_health = own_health

                    if not headless:
                        self.env.render()

                    state = self.reshape_next_state_and_add_to_model_experience(
                        state,
                        last_action,
                        reward,
                        next_state,
                        done
                    )

                    del next_state

                    # for tensorboard
                    episode_reward += reward

                    with self.summary_writer.as_default():
                        tf.summary.scalar('episode reward', reward, step=episode_index)

                print("Episode {}# Reward: {}".format(episode_index, episode_reward))
                print("Training...")
                self.agent.train_on_experience()  # before the next run we fit our model
                print("Done!")

        finally:
            # save the model on quitting training
            self.agent.save_model()

    # add the state and action to the memory to train on once the episode is done
    def reshape_next_state_and_add_to_model_experience(self, state, action, reward, next_state, done):
        next_state = np.reshape(scale(next_state), [1, self.input_size])
        self.agent.add_to_experience(state, action, reward, next_state, done)

        return next_state


def scale(x):
    return x / 255


def get_reward(enemy_health, last_enemy_health, own_health, last_own_health):
    reward = 0

    if enemy_health != last_enemy_health or own_health != last_own_health:
        if enemy_health != base_health or own_health != base_health:

            if last_enemy_health > enemy_health:
                inflicted_damage_reward = (last_enemy_health - enemy_health)
            else:
                inflicted_damage_reward = 0
            # received_damage_penalty = (own_health - last_own_health)
            received_damage_penalty = 0

            # our reward is defined by 'damage I inflict - damage I receive'
            reward = inflicted_damage_reward + received_damage_penalty

            if reward != 0:
                print("Hit enemy for {} reward".format(reward))

            if enemy_health == -1:
                reward = reward + 50  # reward for ko
                print("Enemy KO")
            elif own_health == -1:
                reward = reward - 100  # ... in both ways
                print("Player KO")

    return reward / 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    args = parser.parse_args()

    streetfighter_training = Trainer()
    streetfighter_training.run(args.headless)
