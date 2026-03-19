import numpy as np
import random
from collections import deque, namedtuple
import torch
from .env import WHITE, BLACK

# Single transition tuple
Transition = namedtuple('Transition', (
'state_vec_board', 'state_vec_extra', 'action', 'reward', 'next_state_vec_board', 'next_state_vec_extra', 'done'))


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, device='cpu'):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device

        self.buffer = np.empty(capacity, dtype=object)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

        self.max_priority = 1

    def push_multiple(self, transitions):
        transitions = transitions[:self.capacity] # clip
        for transition in transitions:
            self.push(*transition)

    def push(self, *args):
        if self.pos < self.capacity:
            self.buffer[self.pos] = Transition(*args)
            self.priorities[self.pos] = self.max_priority
            self.pos += 1
        else:
            min_priority_idx = self.priorities.argmin().item()
            self.buffer[min_priority_idx] = Transition(*args)
            self.priorities[min_priority_idx] = self.max_priority

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        batch_size = min(batch_size, self.pos)
        priorities = self.priorities[:self.pos]
        scaled_priorities = priorities ** self.alpha
        priority_sum = scaled_priorities.sum()

        if priority_sum <= 0 or np.isnan(priority_sum):
            probs = np.ones(self.pos, dtype=np.float32) / self.pos
        else:
            probs = scaled_priorities / priority_sum

        indices = np.random.choice(self.pos, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        with torch.no_grad():
            td_errors = td_errors.squeeze()
            # print(self.priorities.shape, indices.shape, td_errors.shape)
            self.priorities[indices] = td_errors + 1e-6
            self.max_priority = self.priorities.max()

    def __len__(self):
        return len(self.buffer)
    """def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)"""


class DQNAgent:
    def __init__(self, model_net, epsilon=0.1, device='cpu', side=WHITE):
        self.net = model_net
        self.epsilon = epsilon
        self.device = device
        self.side = side
        self.net.to(device=device)

    def get_action(self, game, side=None):
        """
        Select action using epsilon-greedy policy.
        """
        if side is None:
            side = self.side
        available_moves = game.get_all_available_moves()
        if not available_moves:
            return None, 0

        # Exploration
        if np.random.random() < self.epsilon:
            return random.choice(available_moves), 0

        # Exploitation
        vec_board, vec_state = game.to_vector()

        # Prepare tensors
        t_board = torch.from_numpy(vec_board).float().to(device=self.device).unsqueeze(0)
        t_state = torch.from_numpy(vec_state).float().to(device=self.device).unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            q_values = self.net(t_board, t_state)  # Shape: [1, n_actions]
            # q_values = q_values.cpu().numpy().flatten()

        # Mask invalid moves with -infinity
        legal_mask = torch.full_like(q_values, -float('inf') if side == WHITE else float('inf'))
        move_to_q_index = {}

        for idx, move in enumerate(available_moves):
            move_id = move.id()
            move_to_q_index[move_id] = idx
            legal_mask[0, move_id] = 0

        q_values += legal_mask

        best_move_idx = torch.argmax(q_values) if side == WHITE else torch.argmin(q_values)
        best_move = available_moves[move_to_q_index[best_move_idx.item()]]
        best_q = q_values[0, best_move_idx].item()  # q-value for the best move

        return best_move, best_q

    def set_epsilon(self, eps):
        self.epsilon = eps
