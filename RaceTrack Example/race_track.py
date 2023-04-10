import numpy as np
import matplotlib.pyplot as plt
import pygame
from matplotlib import colors


actions = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


class RaceTrack:
    def __init__(self):
        self.racetrack = np.full((26, 13), -1)

    def generate_valid_positions(self):
        self.racetrack[6, 6:9] = 0
        self.racetrack[7, 6:12] = 0
        self.racetrack[8, 6:8] = 0
        self.racetrack[8, 9:12] = 0
        self.racetrack[9, 6:8] = 0
        self.racetrack[9, 9] = 0
        self.racetrack[9, 11] = 0
        self.racetrack[10, 6:8] = 0
        self.racetrack[10, 9:12] = 0
        self.racetrack[11, 6:10] = 0
        self.racetrack[11, 11] = 0
        self.racetrack[12, 6:9] = 0
        self.racetrack[13, 6:9] = 0
        self.racetrack[14, 6:9] = 0
        self.racetrack[15, 6:9] = 0
        self.racetrack[16, 6:9] = 0
        self.racetrack[17, 6:10] = 0
        self.racetrack[18, 7:10] = 0
        self.racetrack[19, 7:10] = 0
        self.racetrack[20, 7:10] = 0
        self.racetrack[21, 7:10] = 0
        self.racetrack[22, 7:10] = 0
        self.racetrack[23, 7:10] = 0
        return self.racetrack

    def generate_start_positions(self):
        self.racetrack[24, 7] = 1
        self.racetrack[24, 8] = 1
        self.racetrack[24, 9] = 1
        return self.racetrack

    def generate_finish_position(self):
        self.racetrack[7, 12] = 2
        self.racetrack[8, 12] = 2
        self.racetrack[9, 12] = 2
        self.racetrack[10, 12] = 2
        self.racetrack[11, 12] = 2
        return self.racetrack

    def generate_racetrack(self):
        """
        -1: invalid positions
        0: valid positions
        1: start positions
        2: finish positions
        """
        self.racetrack = self.generate_valid_positions()
        self.racetrack = self.generate_start_positions()
        self.racetrack = self.generate_finish_position()
        return self.racetrack

    def visualize_racetrack(self):
        cmap = colors.ListedColormap(["red", "blue", "green", "yellow"])
        plt.figure(figsize=(6, 6))
        plt.pcolor(self.racetrack[::-1], cmap=cmap, edgecolors='k', linewidths=3)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return None


# generate racetrack
RaceTrackObj = RaceTrack()
race_track = RaceTrackObj.generate_racetrack()
print(race_track)


# visualize racetrack
RaceTrackObj.visualize_racetrack()


class Data:
    def __int__(self):
        self.racetrack = race_track
        self.start_line = np.array([[24, 7], [24, 8], [24, 9]])
        self.finish_line = np.array([[7, 12], [8, 12], [9, 12], [10, 12], [11, 12]])
        self.Q_vals = np.load("data/Q_vals.npy")
        self.C_vals = np.load("data/C_vals.npy")
        self.policy = np.load("data/policy.npy")
        self.rewards = list(np.load("data/rewards.npy"))
        self.epsilon = 0.1
        self.gamma = 1
        self.episode = {
            "S": [],
            "A": [],
            "probs": [],
            "R": [None]
        }


class Env:
    def __init__(self, data):
        self.data = data
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.data.episode = {
            "S": [],
            "A": [],
            "probs": [],
            "R": [None]
        }

    def set_zero(self, numpy_array):
        """Set all the elements of the numpy array to zero.
        """
        numpy_array[:] = 0
        return numpy_array

    def select_random_start_position(self):
        position_x = 24
        random_position_y = np.random.choice([7, 8, 9])
        return (position_x, random_position_y)

    def start(self):
        """Makes the velocity of car zero, and selects any one of the starting
        positions in green.
        """
        state = np.zeros(4, dtype="int")
        state[0], state[1] = self.select_random_start_position()
        return state

    def get_new_state(self, state, action):
        new_state = state.copy()
        new_state[0] = state[0] - state[2]
        new_state[1] = state[1] + state[3]
        new_state[2] = state[2] + action[0]
        new_state[3] = state[3] + action[1]
        return new_state

    def is_finish_line_crossed(self, state, action):
        """
        Returns True if finish line is crosses, False otherwise.
        Finish line is crossed when car reaches any of the a, b, c, d, e grid cells.
        """
        new_state = self.get_new_state(state, action)
        if (new_state[1] == 12) and (new_state[0] in [7, 8, 9, 10, 11]):
            return True
        else:
            return False

    def is_out_of_track(self, state, action):
        """
        Returns True if the car intersects any of the red boundary, False otherwise.
        """
        new_state = self.get_new_state(state, action)
        if race_track[new_state[0], new_state[1]] == -1:
            return True
        else:
            return False

    def step(self, state, action):
        reward = -1
        self.data.episode["A"].append(action)
        if self.is_finish_line_crossed(state, action):
            new_state = self.get_new_state(state, action)
            self.data.episode["R"].append(reward)
            self.data.episode["S"].append(new_state)
            self.step_count += 1
            return None, new_state

        elif self.is_out_of_track(state, action):
            new_state = self.start()

        else:
            new_state = self.get_new_state(state, action)

        self.data.episode["R"].append(reward)
        self.data.episode["S"].append(new_state)
        self.step_count += 1

        return reward, new_state


class Agent:
    def __int__(self):
        pass

    def get_indices_of_valid_actions(self, velocity):
        indices_of_valid_acts = list()
        for idx, possible_action in enumerate(actions):
            new_velocity = np.add(velocity, possible_action)
            if (new_velocity[0] <= 5 and new_velocity[0] >=0) and (new_velocity[1] <=5 and new_velocity[1] >= 0):
                indices_of_valid_acts.append(idx)
        indices_of_valid_acts = np.array(indices_of_valid_acts)
        return indices_of_valid_acts

    def map_to_one_dimension(self, action):
        for idx, possible_action in enumerate(actions):
            if list(action) == list(possible_action):
                return idx

    def map_to_two_dimension(self, idx_of_action):
        return actions[idx_of_action]

    def get_action(self, state, policy):
        """Returns next action given the state following a policy.
        """
        return self.map_to_two_dimension(policy(state, self.get_indices_of_valid_actions(state[2:4])))


class Visualizer:
    """The visualizer takes the state of the system and creates apygame window
    to visualize the current location of the agent on top of the racetrack.
    """
    def __int__(self, data):
        self.data = data
        self.window = False
        self.cell_edge = 9
        self.width = 26 * self.cell_edge
        self.height = 13 * self.cell_edge

    def create_window(self):
        """Creates window and assigns self.display variable
        """
        # self.display = pygame.display.set_mode((self.width, self.height))
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack")

    def setup(self):
        self.create_window()
        self.window = True

    def close_window(self):
        self.window = False
        pygame.quit()

    def draw(self, state=np.array([])):
        self.display.fill(0)
        for i in range(26):
            for j in range(13):
                if self.data.racetrack[i, j] != -1:
                    color = (0, 0, 0)
                    if self.data.racetrack[i, j] == 0:
                        color = (0, 0, 255)
                    elif self.data.racetrack[i, j] == 1:
                        color = (0, 255, 0)
                    elif self.data.racetrack[i, j] == 2:
                        color = (255, 165, 0)
                    pygame.draw.rect(self.display, color, ((j * self.cell_edge, i * self.cell_edge), (self.cell_edge, self.cell_edge)), 1)

        if len(state) > 0:
            pygame.draw.rect(self.display, (255, 0, 0), ((state[1] * self.cell_edge, state[0] * self.cell_edge), (self.cell_edge, self.cell_edge)), 1)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
                self.close_window()
                return "stop"
            elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_SPACE):
                self.loop = False

        return None

    def visualize_racetrack(self, state=np.array([])):
        if not self.window:
            self.setup()
        self.loop = True
        while(self.loop):
            ret = self.draw(state)
            if ret is not None:
                return ret


# initialize environment
env = Env()
# initialize agent
agent = Agent()


class OffPolicyMonteCarloControl:
    def __init__(self, data):
        """Intitialize for all states, actions, etc."""
        self.data = data
        for i in range(26):
            for j in range(13):
                if self.data.racetrack[i, j] != -1:
                    for k in range(6):
                        for l in range(6):
                            self.data.policy[i, j, k, l] = np.argmax(self.data.Q_vals[i, j, k, l])

    def determine_probability_behaviour(self, state, action, possible_actions):
        best_action = self.data.policy[tuple(state)]
        num_actions = len(possible_actions)

        if best_action in possible_actions:
            if action == best_action:
                prob = 1 - self.data.epsilon + self.data.epsilon / num_actions
            else:
                prob = self.data.epsilon / num_actions
        else:
            prob = 1 / num_actions

        self.data.episode["probs"].append(prob)

    def generate_target_policy_action(self, state, possible_actions):
        """Returns target policy action; takes state and return an action
        using this policy.
        """
        if self.data.policy[tuple(state)] in possible_actions:
            action = self.data.policy[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        self.determine_probability_behaviour(state, action, possible_actions)

        return action

    def generate_behavioural_policy_action(self, state, possible_actions):
        """Returns behavioural policy action which would be epsilon-greedy pi policy;
        takes state and returns an action using this epsilon-greedy pi policy."""
        if np.random.rand() > self.data.epsilon and self.data.policy[tuple(state)] in possible_actions:
            action = self.data.policy[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        self.determine_probability_behaviour(state, action, possible_actions)

        return action

    def evaluate_target_policy(self):
        env.reset()
        state = env.start()
        self.data.episode["S"].append(state)
        rew = -1
        while rew is not None:
            action = agent.get_action(state, self.generate_target_policy_action)
            rew, state = env.step(state, action)

        self.data.rewards.append(sum(self.data.episode["R"][1:]))

    def save_your_work(self):
        self.data.save_Q_vals()
        self.data.save_C_vals()
        self.data.save_policy()
        self.data.save_rewards()

    def control(self, env, agent):
        """Performs MC control using episode list"""
        env.reset()
        state= env.start()
        self.data.episode["S"].append(state)
        rew = - 1
        while rew is not None:
            action = agent.get_action(state, self.generate_behavioural_policy_action)
            rew, state = env.step(state, action)

        G = 0
        W = 1
        T = env.step_count

        for t in range(T - 1, -1, -1):
            G = data.gamma * G + self.data.episode["R"][t+1]
            S_t = tuple(self.data.episode["S"][t])
            A_t = agent.map_to_one_dimension(self.data.episode["A"][t])

            S_list = list(S_t)
            S_list.append(A_t)
            SA = tuple(S_list)

            self.data.C_vals[SA] += W
            self.data.Q_vals[SA] += (W * (G - self.data.Q_vals[SA])) / (self.data.C_vals[SA])
            self.data.policy[S_t] = np.argmax(self.data.Q_vals[S_t])
            if A_t != self.data.policy[S_t]:
                break
            W /= self.data.episode["probs"][t]

    def plot_rewards(self):
        ax, fig = plt.subplots(figsize=(30, 15))
        x = np.arange(1, len(self.data.rewards) + 1)
        plt.plot(x * 10, self.data.rewards, linewidth=0.5, color="#BB8FCE")
        plt.xlabel("Episode number", size=20)
        plt.ylabel("Reward", size=20)
        plt.title("Plot of Reward vs Episode Number", size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig("Reward_Graph.png")
        plt.close()

