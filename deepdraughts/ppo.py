import numpy as np
import torch
from torch.distributions import Categorical
from .env import WHITE, BLACK


def flip_board_perspective(vec_board):
    """
    Rotating 2D board that should allow for training black and white
    """
    flipped = np.zeros_like(vec_board)

    # flipping channels not sure if i do it correctly
    flipped[0] = vec_board[2]
    flipped[1] = vec_board[3]
    flipped[2] = vec_board[0]
    flipped[3] = vec_board[1]

    flipped = np.rot90(flipped, k=2, axes=(1, 2))

    return flipped.copy()


class RolloutBuffer:
    def __init__(self):
        self.boards = []
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []
        self.advantages = []
        self.returns = []

    def clear(self):
        del self.boards[:]
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]
        del self.masks[:]
        del self.advantages[:]
        del self.returns[:]


class PPOAgent:
    def __init__(self, model_net, device='cpu', side=WHITE):
        self.net = model_net
        self.device = device
        self.side = side

    def get_action(self, game, side=None, deterministic=False):
        if side is None:
            side = self.side

        available_moves = game.get_all_available_moves()
        if not available_moves:
            # when no move availble return None
            return None, 0.0, 0.0, None

        vec_board, vec_state = game.to_vector()

        # when playing black we rotate
        if game.current_player == BLACK:
            vec_board = flip_board_perspective(vec_board)

        t_board = torch.from_numpy(vec_board).float().to(device=self.device).unsqueeze(0)
        t_state = torch.from_numpy(vec_state).float().to(device=self.device).unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(t_board, t_state)

        logits = logits.squeeze(0).cpu().numpy()
        value = value.item()

        # illegal moves masking
        mask = np.zeros_like(logits, dtype=bool)
        for m in available_moves:
            mask[m.id()] = True

        logits[~mask] = -1e9  # it should "block" taking move

        t_logits = torch.from_numpy(logits).float().to(self.device)
        probs = torch.softmax(t_logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action_idx = torch.argmax(probs).item()
            log_prob = dist.log_prob(torch.tensor(action_idx).to(self.device)).item()
        else:
            action_tensor = dist.sample()
            action_idx = action_tensor.item()
            log_prob = dist.log_prob(action_tensor).item()

        best_move = next((m for m in available_moves if m.id() == action_idx), available_moves[0])

        return best_move, log_prob, value, mask