import argparse

import numpy as np
import retro
from Agent import Agent


def action_to_array(action, n):
    # define some cool streetfighter actions here
    # action_set = {
    #     # ["B", "A",    "MODE", "START",    "UP", "DOWN", "LEFT", "RIGHT",      "C", "Y", "X", "Z"]
    #     # [med kick, light kick, -, -, -, -, -, -, -, hard kick, med punch, light punch, hard punch]
    #     0:  np.array([0, 0,     0, 0,   0, 1, 0, 0,     0, 0, 0, 0]),
    #     1:  np.array([0, 0,     0, 0,   0, 1, 0, 1,     0, 0, 0, 0]),
    #     2:  np.array([0, 0,     0, 0,   0, 0, 0, 1,     1, 0, 0, 0]),
    #
    #     3:  np.array([1, 0,     0, 0,   1, 0, 1, 0,     0, 0, 0, 0]),
    #     4:  np.array([0, 1,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #     5:  np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #
    #     6:  np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #     7:  np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #
    #     8:  np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #     9:  np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #     10: np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    #     11: np.array([0, 0,     0, 0,   0, 0, 0, 0,     0, 0, 0, 0]),
    # }
    # return action_set[action]
    action_array = np.zeros(n)
    action_array[action] = 1
    return action_array


class Trainer:
    def __init__(self):
        self.env = retro.make(
            'StreetFighterIISpecialChampionEdition-Genesis',
            scenario='scenario',
            obs_type=retro.Observations.RAM
        )

        self.sample_batch_size = 32
        self.episodes = 10000
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.agent = Agent(self.input_size, self.output_size)

    def run(self, headless):
        # dirty af but idgaf
        base_health = 176

        try:
            for episode_index in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.input_size])

                done = False
                total_reward = 0
                index = 0

                last_enemy_health = base_health
                last_own_health = base_health

                while not done:
                    action = self.agent.get_next_action(state)

                    next_state, _, done, mem_info = self.env.step(action_to_array(action, self.output_size))

                    enemy_health = mem_info['enemy_health']
                    own_health = mem_info['health']

                    reward = 0

                    if enemy_health != last_enemy_health or own_health != last_own_health:

                        if enemy_health != base_health or own_health != base_health:
                            damage_reward = (last_enemy_health - enemy_health) / base_health
                            avoidance_reward = (own_health - last_own_health) / base_health

                            reward = damage_reward + avoidance_reward

                            if enemy_health == -1:
                                reward = reward * 2  # double reward for ko
                            elif own_health == -1:
                                reward = reward * 2

                            last_enemy_health = enemy_health
                            last_own_health = own_health

                    if not headless:
                        self.env.render()

                    next_state = np.reshape(next_state, [1, self.input_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    total_reward = total_reward + reward

                    index += 1

                    state = next_state
                print("Episode {}# Score: {} Index: {}".format(episode_index, total_reward, index))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    args = parser.parse_args()

    streetfighter_training = Trainer()
    streetfighter_training.run(args.headless)
