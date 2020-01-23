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

        start = np.zeros(self.output_size)
        start[5] = 1
        self.start = start

        self.noop = np.zeros(self.output_size)
        self.frameskip_enabled = True
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

                # this is for frameskip
                frame = -1

                while not done:
                    frame += 1

                    # predict the best action for the current state s
                    action = self.agent.get_action_for(state)

                    # apply the predicted action to the sate and receive next_state
                    next_state, _, done, ram = self.env.step(action_to_array(action, self.output_size))

                    # ------------------------------------------------------------------------------------------------ #
                    # skip n frames so that the observed s' is acutally a consequence of s(a)
                    if self.frameskip_enabled:
                        for _ in range(self.frameskip):
                            if not done:
                                next_state, _, done, ram = self.env.step(self.noop)

                    enemy_health = ram['enemy_health']
                    own_health = ram['health']

                    # get the reward for this s' for t+n
                    reward = get_reward(enemy_health, last_enemy_health, own_health, last_own_health)

                    # check if the round is over...
                    if own_health <= -1 or enemy_health <= -1:  # this means the round is over
                        for _ in range(5):  # skip some frames so that scores come in
                            _, _, done, ram = self.env.step(self.start)

                        print("Round over, {}:{} got {} reward without KO".format(
                            ram['matches_won'],
                            ram['enemy_matches_won'],
                            episode_reward
                        ))

                        # remember these for remembering below, we first need to find out if done is true or not
                        start_state = state
                        skipped_to_state = next_state

                        # we step thru the waiting screen, in case it's the second win we want the episode to be done
                        while True:
                            _, _, done, ram = self.env.step(self.start)
                            if (ram['enemy_health'] == base_health and ram['health'] == base_health) or done:
                                break

                        # add this terminal state (state, action, state', reward and DONE) to the models experience
                        state = self.remember(start_state, action, reward, skipped_to_state, done)

                        # we end this training session and start the next frame with new state and reset health
                        last_enemy_health = base_health
                        last_own_health = base_health

                    # ...or if it was just another frame
                    else:
                        # get ready for the next frame
                        last_enemy_health = enemy_health
                        last_own_health = own_health

                        if not headless:
                            self.env.render()

                        # add the frame (state, action, state', reward) to the models experience
                        state = self.remember(state, action, reward, next_state, done)

                        # log some metrics to tensorboard
                        episode_reward += reward
                        with self.summary_writer.as_default():
                            tf.summary.scalar('episode reward', reward, step=episode_index)

                    del next_state

                print("Episode {}# Reward: {}".format(episode_index, episode_reward))
                print("Training...")
                self.agent.train_on_experience()  # before the next episode we fit our model
                print("Done!")

        finally:
            # save the model on quitting training
            self.agent.save_model()

    # add the state and action to the memory to train on once the episode is done
    def remember(self, state, action, reward, next_state, done):
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

    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    args = parser.parse_args()

    streetfighter_training = Trainer()
    streetfighter_training.run(args.headless)
