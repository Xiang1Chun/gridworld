import numpy as np

class QLearning:
    def __init__(self):
        self.grid = None
        self.start = None
        self.end = None
        self.blocks = None
        self.q_table = None

    def setup(self, grid, start, end, blocks):
        self.grid = grid
        self.start = tuple(start)
        self.end = tuple(end)
        self.blocks = [tuple(block) for block in blocks]

        self.q_table = np.zeros((len(self.grid), len(self.grid[0]), 4))

    def train(self, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for episode in range(episodes):
            state = self.start

            while state != self.end:
                if np.random.rand() < epsilon:
                    action_idx = np.random.randint(len(actions))
                else:
                    action_idx = np.argmax(self.q_table[state])

                next_state = (state[0] + actions[action_idx][0], state[1] + actions[action_idx][1])

                if (next_state[0] < 0 or next_state[0] >= len(self.grid) or
                        next_state[1] < 0 or next_state[1] >= len(self.grid[0]) or
                        next_state in self.blocks):
                    reward = -100
                    next_state = state
                elif next_state == self.end:
                    reward = 100
                else:
                    reward = -1

                self.q_table[state][action_idx] = (1 - alpha) * self.q_table[state][action_idx] + alpha * (reward + gamma * np.max(self.q_table[next_state]))

                state = next_state

    def get_optimal_path(self):
        path = [self.start]
        state = self.start
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while state != self.end:
            action_idx = np.argmax(self.q_table[state])
            next_state = (state[0] + actions[action_idx][0], state[1] + actions[action_idx][1])
            path.append(next_state)
            state = next_state

        return path
