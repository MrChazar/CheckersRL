# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import datetime
import copy


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)


class DQNNet(nn.Module):
    """DQN Network: Input (Board, State) -> Output (Q-values for all actions)"""

    def __init__(self, nsize, n_states, n_actions):
        super(DQNNet, self).__init__()

        self.nsize = nsize
        self.board_width = nsize
        self.board_height = nsize
        self.n_states = n_states
        self.n_actions = n_actions

        # Feature Extractor (CNN)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # State processing
        self.st_fc1 = nn.Linear(n_states, 128)
        self.st_bn1 = nn.BatchNorm1d(128)

        # linear
        #self.fc_common = nn.Linear(128 * nsize * nsize + 64, 512)
        #self.fc_q_head = nn.Linear(512, n_actions)

        # ending layers of standard DQN above (commented out)
        # ending layers implementing dueling DQN belowself.
        self.fc_common = nn.Linear(64 * nsize * nsize + 128, 512)
        self.fc_common_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)

        # Value stream
        self.fc_value = nn.Linear(512, 256)
        self.fc_value_bn = nn.BatchNorm1d(256)
        self.value_head = nn.Linear(256, 1)

        # Advantage stream
        self.fc_adv = nn.Linear(512, 256)
        self.fc_adv_bn = nn.BatchNorm1d(256)
        self.adv_head = nn.Linear(256, n_actions)

        _initialize_weights(self)

    def forward(self, vec_board, vec_state):

        if vec_board.dim() == 3:
            vec_board = vec_board.unsqueeze(0)

        if vec_state.dim() == 1:
            vec_state = vec_state.unsqueeze(0)

        assert vec_board.shape[1] == 4, f"Expected board shape (B, 4, H, W), got {vec_board.shape}"
        assert vec_state.shape[1] == self.n_states, f"Expected state shape (B, {self.n_states}), got {vec_state.shape}"

        # CNN Path
        x = F.relu(self.bn1(self.conv1(vec_board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, start_dim=1) # Flatten

        # State Path
        y = F.relu(self.st_bn1(self.st_fc1(vec_state)))

        # Merge
        combined = torch.cat((x, y), 1)

        # Q-Values
        feat = F.relu(self.fc_common_bn(self.fc_common(combined)))
        feat = self.dropout(feat)

        # Value stream
        value = F.relu(self.fc_value_bn(self.fc_value(feat)))
        value = self.value_head(value)  # shape: (batch, 1)

        # Advantage stream
        adv = F.relu(self.fc_adv_bn(self.fc_adv(feat)))
        adv = self.adv_head(adv)  # shape: (batch, n_actions)

        # Combine streams
        q_values = value + adv - adv.mean(dim=1, keepdim=True)

        return q_values

class Model:
    """DQN Model Handler with Target Network"""

    def __init__(self, env_args, name="dqn_default", device='cpu', l2_const=1e-4, recursive_steps=4):
        nsize, _, n_states, n_actions = env_args
        self.nsize = nsize
        self.n_states = n_states
        self.n_actions = n_actions
        self.name = name
        self.device = device
        self.l2_const = l2_const

        # Policy Network (Training)
        self.policy_net = DQNNet(nsize, n_states, n_actions)
        # Target Network (Stable targets)
        self.target_net = DQNNet(nsize, n_states, n_actions)

        self.policy_net = self.policy_net.to(device=self.device)
        self.target_net = self.target_net.to(device=self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly !!!!

        self.optimizer = optim.Adam(self.policy_net.parameters(), weight_decay=self.l2_const, )

    def sync_target_network(self):
        """Copy weights from policy net to target net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Target Network Synced.")

    def train_step(self, batch_data, gamma, lr, weights, n_steps):
        """
        Perform one Double DQN training step.

        Expected batch_data:
            (
                b_board,
                b_state,
                b_action,
                b_reward,
                b_next_board,
                b_next_state,
                b_done,
                b_next_legal_mask
            )

        b_next_legal_mask shape:
            [batch_size, n_actions]
            True  -> legal action
            False -> illegal action
        """

        (
            b_board,
            b_state,
            b_action,
            b_reward,
            b_next_board,
            b_next_state,
            b_done,
            b_next_legal_mask,
        ) = batch_data

        self.policy_net.train()
        self.target_net.eval()

        # Convert batch to tensors
        state_board = torch.as_tensor(b_board, dtype=torch.float32, device=self.device)
        state_extra = torch.as_tensor(b_state, dtype=torch.float32, device=self.device)

        action = torch.as_tensor(b_action, dtype=torch.long, device=self.device)
        if action.dim() == 1:
            action = action.unsqueeze(1)  # [B, 1]

        reward = torch.as_tensor(b_reward, dtype=torch.float32, device=self.device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)  # [B, 1]

        next_board = torch.as_tensor(b_next_board, dtype=torch.float32, device=self.device)
        next_extra = torch.as_tensor(b_next_state, dtype=torch.float32, device=self.device)

        done = torch.as_tensor(b_done, dtype=torch.float32, device=self.device)
        if done.dim() == 1:
            done = done.unsqueeze(1)  # [B, 1]

        next_legal_mask = torch.as_tensor(
            b_next_legal_mask,
            dtype=torch.bool,
            device=self.device
        )  # [B, n_actions]

        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        if weights.dim() == 1:
            weights = weights.unsqueeze(1)  # [B, 1]

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # Current Q(s, a)
        q_values = self.policy_net(state_board, state_extra)  # [B, n_actions]
        q_val = q_values.gather(1, action)  # [B, 1]

        # Double DQN target
        with torch.no_grad():
            # Important if policy_net uses BatchNorm:
            # use eval mode while selecting next actions, so BN stats are not updated
            was_training = self.policy_net.training
            self.policy_net.eval()

            next_q_online = self.policy_net(next_board, next_extra)  # [B, n_actions]

            if was_training:
                self.policy_net.train()

            # Mask illegal actions in next_state
            next_q_online_masked = next_q_online.masked_fill(
                ~next_legal_mask,
                -float("inf")
            )

            # WALIDACJA: sprawdź czy są nielegalne akcje
            has_legal_moves = next_legal_mask.any(dim=1)
            if not has_legal_moves.all():
                # Jeśli nie ma legalnych ruchów, ustaw Q-value na 0
                next_q_online_masked[~has_legal_moves] = 0.0

            # Select best legal action according to online net
            next_actions = next_q_online_masked.argmax(dim=1, keepdim=True)  # [B, 1]

            # Evaluate selected action using target net
            next_q_target = self.target_net(next_board, next_extra)  # [B, n_actions]
            next_q_value = next_q_target.gather(1, next_actions)  # [B, 1]

            # If b_reward is already n-step return, gamma ** n_steps is correct.
            # If b_reward is only immediate reward, use gamma instead.
            expected_q_val = reward + (gamma ** n_steps) * next_q_value * (1.0 - done)

        td_errors = q_val - expected_q_val  # [B, 1]

        # Huber loss per sample
        loss_per_sample = F.smooth_l1_loss(
            q_val,
            expected_q_val,
            reduction="none"
        )  # [B, 1]

        # Apply importance sampling weights, e.g. from prioritized replay
        loss = (weights * loss_per_sample).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=5.0
        )

        self.optimizer.step()

        return loss.item(), next_q_online.detach(), td_errors.detach()

    def save(self, savepath, epoch, is_best=False):

        torch.save({
            'nsize': self.nsize,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'name': self.name,
            'n_epoch': epoch,
            'device': self.device,
            'l2_const': self.l2_const,
            'model': self.policy_net.state_dict(),
            'target_model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, savepath)

    @classmethod
    def load(cls, model_file, device='cpu'):
        # Implementation similar to the original code, adapted for DQN
        model_params = torch.load(model_file, map_location=torch.device(device=device))
        env_args = (model_params['nsize'], model_params['device'], model_params['n_states'], model_params['n_actions'])
        model = Model(env_args, model_params['name'], device, model_params['l2_const'])
        model.policy_net.load_state_dict(model_params['model'])
        if 'target_model' in model_params:
            model.target_net.load_state_dict(model_params['target_model'])  # Sync target on load
        else:
            model.target_net.load_state_dict(model_params['model'])  # Sync target on load
        model.optimizer.load_state_dict(model_params['optimizer'])
        return model


from collections import OrderedDict


def copy_net_to_cpu(net):
    """
    Tworzy kopię modelu na CPU, zachowując typ modelu.
    Obsługuje:
    - DQNNet
    - RecursiveDQNNet
    """

    if isinstance(net, DQNNet):
        use_net = DQNNet(
            net.nsize,
            net.n_states,
            net.n_actions
        )
    #
    # elif isinstance(net, RecursiveDQNNet):
    #     use_net = RecursiveDQNNet(
    #         net.nsize,
    #         net.n_states,
    #         net.n_actions,
    #         recursive_steps=net.recursive_steps
    #     )

    else:
        raise TypeError(f"Unsupported model type: {type(net).__name__}")

    state_dict = OrderedDict(
        (k, v.detach().cpu().clone())
        for k, v in net.state_dict().items()
    )

    use_net.load_state_dict(state_dict)
    use_net.to("cpu")
    use_net.eval()

    return use_net