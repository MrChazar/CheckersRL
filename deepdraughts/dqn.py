import numpy as np
import random
from collections import deque, namedtuple
import torch
from .env import WHITE, BLACK

# Single transition tuple
Transition = namedtuple('Transition', (
'state_vec_board', 'state_vec_extra', 'action', 'reward', 'next_state_vec_board', 'next_state_vec_extra', 'done'))


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = Transition(*args)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

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

        q_values = q_values.cpu().numpy().flatten()

        # Mask invalid moves with -infinity
        #legal_ids = [m.id() for m in available_moves]

        if side == WHITE:
            best_q = -float('inf')
        else:
            best_q = float('inf')
        best_move = available_moves[0]

        # Simple loop to find max Q among legal moves
        for move in available_moves:
            mid = move.id()
            if (q_values[mid] > best_q and side == WHITE) or (q_values[mid] < best_q and side == BLACK):
                best_q = q_values[mid]
                best_move = move

        return best_move, best_q

    def set_epsilon(self, eps):
        self.epsilon = eps
