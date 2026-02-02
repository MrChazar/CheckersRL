import numpy as np
import random
from collections import deque, namedtuple
import torch

# Single transition tuple
Transition = namedtuple('Transition', (
'state_vec_board', 'state_vec_extra', 'action', 'reward', 'next_state_vec_board', 'next_state_vec_extra', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, model_net, epsilon=0.1, use_gpu=False):
        self.net = model_net
        self.epsilon = epsilon
        self.use_gpu = use_gpu

    def get_action(self, game):
        """
        Select action using epsilon-greedy policy.
        """
        available_moves = game.get_all_available_moves()
        if not available_moves:
            return None, 0

        # Exploration
        if np.random.random() < self.epsilon:
            return random.choice(available_moves), 0

        # Exploitation
        vec_board, vec_state = game.to_vector()

        # Prepare tensors
        if self.use_gpu:
            t_board = torch.from_numpy(vec_board).cuda().float().unsqueeze(0)
            t_state = torch.from_numpy(vec_state).cuda().float().unsqueeze(0)
        else:
            t_board = torch.from_numpy(vec_board).float().unsqueeze(0)
            t_state = torch.from_numpy(vec_state).float().unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            q_values = self.net(t_board, t_state)  # Shape: [1, n_actions]

        q_values = q_values.cpu().numpy().flatten()

        # Mask invalid moves with -infinity
        legal_ids = [m.id() for m in available_moves]

        best_q = -float('inf')
        best_move = available_moves[0]

        # Simple loop to find max Q among legal moves
        for move in available_moves:
            mid = move.id()
            if q_values[mid] > best_q:
                best_q = q_values[mid]
                best_move = move

        return best_move, best_q

    def set_epsilon(self, eps):
        self.epsilon = eps
