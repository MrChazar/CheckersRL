import numpy as np
import random
from collections import deque, namedtuple
import torch

from deepdraughts.env import WHITE

# Single transition tuple
Transition = namedtuple('Transition', (
'state_vec_board', 'state_vec_extra', 'action', 'reward', 'next_state_vec_board', 'next_state_vec_extra', 'done', 'b_next_legal_mask'))


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

    def get_action(self, game, side=None):
        """
        Select action using epsilon-greedy policy.
        Assumption: Q-values are from the perspective of the current player,
        so the agent always chooses argmax among legal actions.
        """
        available_moves = game.get_all_available_moves()
        if not available_moves:
            return None, 0

        # Exploration
        if np.random.random() < self.epsilon:
            return random.choice(available_moves), 0

        vec_board, vec_state = game.to_vector()

        t_board = torch.as_tensor(vec_board, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_state = torch.as_tensor(vec_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        was_training = self.net.training
        self.net.eval()

        with torch.no_grad():
            q_values = self.net(t_board, t_state)[0]  # [n_actions]

            legal_move_by_id = {}
            legal_ids = []

            for move in available_moves:
                move_id = move.id()

                if move_id < 0 or move_id >= q_values.shape[0]:
                    raise ValueError(
                        f"Illegal move_id={move_id}. Expected range: 0..{q_values.shape[0] - 1}"
                    )

                legal_move_by_id[move_id] = move
                legal_ids.append(move_id)

            legal_ids = torch.as_tensor(legal_ids, dtype=torch.long, device=self.device)

            legal_q_values = q_values[legal_ids]
            best_local_idx = torch.argmax(legal_q_values).item() if side == WHITE else torch.argmin(legal_q_values).item()

            best_move_id = legal_ids[best_local_idx].item()
            best_move = legal_move_by_id[best_move_id]
            best_q = q_values[best_move_id].item()

        if was_training:
            self.net.train()

        return best_move, best_q

    def set_epsilon(self, eps):
        self.epsilon = eps

