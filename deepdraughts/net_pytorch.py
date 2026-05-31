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
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # State processing
        self.st_fc1 = nn.Linear(n_states, 64)

        # linear
        #self.fc_common = nn.Linear(128 * nsize * nsize + 64, 512)
        #self.fc_q_head = nn.Linear(512, n_actions)

        # ending layers of standard DQN above (commented out)
        # ending layers implementing dueling DQN below
        self.fc_common = nn.Linear(128 * nsize * nsize + 64, 512)

        # Value stream
        self.fc_value = nn.Linear(512, 256)
        self.value_head = nn.Linear(256, 1)

        # Advantage stream
        self.fc_adv = nn.Linear(512, 256)
        self.adv_head = nn.Linear(256, n_actions)

        _initialize_weights(self)

    def forward(self, vec_board, vec_state):
        if len(vec_board.shape) == 3:
            vec_board = torch.unsqueeze(vec_board, 0)
            vec_state = torch.unsqueeze(vec_state, 0)

        # CNN Path
        x = F.relu(self.bn1(self.conv1(vec_board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten

        # State Path
        y = F.relu(self.st_fc1(vec_state))

        # Merge
        combined = torch.cat((x, y), 1)

        # Q-Values
        #feat = F.relu(self.fc_common(combined))
        #q_values = self.fc_q_head(feat)

        # Q-Values part implementation in standard DQN above (commented out)
        # Q-Values part implementation in dueling DQN part below
        feat = F.relu(self.fc_common(combined))

        # Value stream
        value = F.relu(self.fc_value(feat))
        value = self.value_head(value)  # shape: (batch, 1)

        # Advantage stream
        adv = F.relu(self.fc_adv(feat))
        adv = self.adv_head(adv)  # shape: (batch, n_actions)

        # Combine streams
        q_values = value + (adv - adv.mean(dim=1, keepdim=True))

        return q_values

class RecursiveDQNNet(nn.Module):
    """
    Recursive DQN Network:
    Input:  (Board, State)
    Output: Q-values for all actions

    Czyli tak samo jak Twój DQNNet:
        forward(vec_board, vec_state) -> [batch, n_actions]

    Różnica:
        zamiast jednego przejścia przez końcówkę sieci,
        model kilka razy aktualizuje latent z.
    """

    def __init__(self, nsize, n_states, n_actions, recursive_steps=4):
        super(RecursiveDQNNet, self).__init__()

        self.nsize = nsize
        self.board_width = nsize
        self.board_height = nsize
        self.n_states = n_states
        self.n_actions = n_actions
        self.recursive_steps = recursive_steps

        # Feature Extractor (CNN) — tak jak w Twoim DQN
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # State processing — tak jak w Twoim DQN
        self.st_fc1 = nn.Linear(n_states, 64)

        # Wspólna reprezentacja pozycji
        self.fc_common = nn.Linear(128 * nsize * nsize + 64, 512)

        # Rekurencyjna poprawa latentu.
        # base_feat = stała informacja o pozycji
        # z = zmienny latent "namysłu"
        self.refiner = nn.GRUCell(
            input_size=512,
            hidden_size=512
        )

        # Dueling DQN heads — jak u Ciebie
        # self.fc_value = nn.Linear(512, 256)
        # self.value_head = nn.Linear(256, 1)
        #
        # self.fc_adv = nn.Linear(512, 256)
        # self.adv_head = nn.Linear(256, n_actions)

        # Standard DQN head
        self.fc_q = nn.Linear(512, 256)
        self.q_head = nn.Linear(256, n_actions)

        _initialize_weights(self)

    def _encode(self, vec_board, vec_state):
        """
        Zamienia planszę i dodatkowy stan na wspólny feature vector.
        """

        if len(vec_board.shape) == 3:
            vec_board = torch.unsqueeze(vec_board, 0)

        if len(vec_state.shape) == 1:
            vec_state = torch.unsqueeze(vec_state, 0)

        # CNN Path
        x = F.relu(self.bn1(self.conv1(vec_board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        # State Path
        y = F.relu(self.st_fc1(vec_state))

        # Merge
        combined = torch.cat((x, y), dim=1)

        # Bazowa reprezentacja pozycji
        base_feat = F.relu(self.fc_common(combined))

        return base_feat

    def _q_from_latent(self, z):
        """
        latent z -> Q-values dla wszystkich akcji
        """

        # # Value stream
        # value = F.relu(self.fc_value(z))
        # value = self.value_head(value)  # [batch, 1]
        #
        # # Advantage stream
        # adv = F.relu(self.fc_adv(z))
        # adv = self.adv_head(adv)  # [batch, n_actions]
        #
        # # Combine streams
        # q_values = value + (adv - adv.mean(dim=1, keepdim=True))

        q = F.relu(self.fc_q(z))
        q_values = self.q_head(q)

        return q_values

    def forward(self, vec_board, vec_state, steps=None, return_all_q=False):
        """
        Standardowo:
            q_values = model(vec_board, vec_state)

        Opcjonalnie:
            q_per_step = model(
                vec_board,
                vec_state,
                return_all_q=True
            )

        shapes:
            q_values:   [batch, n_actions]
            q_per_step: [batch, steps, n_actions]
        """

        if steps is None:
            steps = self.recursive_steps
        steps = int(steps)

        base_feat = self._encode(vec_board, vec_state)

        # Początkowy latent.
        # To jest pierwsza reprezentacja pozycji po encoderze.
        z = base_feat

        q_values_per_step = []

        for _ in range(steps):
            # Rekurencyjne poprawienie latentu.
            # base_feat jest cały czas tym samym wejściem,
            # z jest aktualizowanym stanem "namysłu".
            z = self.refiner(base_feat, z)

            q_values = self._q_from_latent(z)
            q_values_per_step.append(q_values)

        if return_all_q:
            return torch.stack(q_values_per_step, dim=1)

        return q_values_per_step[-1]

class Model():
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
        self.policy_net = RecursiveDQNNet(nsize, n_states, n_actions, recursive_steps= recursive_steps)
        # Target Network (Stable targets)
        self.target_net = RecursiveDQNNet(nsize, n_states, n_actions, recursive_steps= recursive_steps)

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
        Perform a single training step using a batch of data.
        batch_data: (b_board, b_state, b_action, b_reward, b_next_board, b_next_state, b_done)
        """
        b_board, b_state, b_action, b_reward, b_next_board, b_next_state, b_done = batch_data

        # Convert to tensors
        state_board = torch.from_numpy(b_board).float()
        state_extra = torch.from_numpy(b_state).float()
        action = torch.from_numpy(b_action).long().unsqueeze(1)  # [Batch, 1]
        reward = torch.from_numpy(b_reward).float().unsqueeze(1)
        next_board = torch.from_numpy(b_next_board).float()
        next_extra = torch.from_numpy(b_next_state).float()
        done = torch.from_numpy(b_done).float().unsqueeze(1)

        state_board, state_extra = state_board.to(device=self.device), state_extra.to(device=self.device)
        action, reward = action.to(device=self.device), reward.to(device=self.device)
        next_board, next_extra = next_board.to(device=self.device), next_extra.to(device=self.device)
        done = done.to(device=self.device)

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # Calculate current Q(s, a)
        # We gather the Q-value corresponding to the taken action
        q_values = self.policy_net(state_board, state_extra)
        q_val = q_values.gather(1, action)

        # Calculate Max Q(s', a') from Target Net
        #with torch.no_grad():
        #    # it corespondends to bellman equation i highly recommend to check presentation
        #    next_q_values = self.target_net(next_board, next_extra)
        #    next_q_max = next_q_values.max(1)[0].unsqueeze(1)
        #    expected_q_val = reward + (gamma * next_q_max * (1 - done))
        with torch.no_grad():
            # Step 1: choose best action from online net
            next_q_online = self.policy_net(next_board, next_extra)
            next_actions = next_q_online.argmax(1, keepdim=True)

            # Step 2: evaluate that action with target net
            next_q_target = self.target_net(next_board, next_extra)
            next_q_value = next_q_target.gather(1, next_actions)

            # Step 3: compute expected Q
            expected_q_val = reward + (gamma ** n_steps) * next_q_value * (1 - done)

        # Loss (MSE) between policy and target net
        # we will use it for gradient descent !
        #loss = F.smooth_l1_loss(q_val, expected_q_val)
        td_errors = q_val - expected_q_val

        # Huber loss per sample (no reduction!)
        loss_per_sample = F.smooth_l1_loss(q_val, expected_q_val, reduction='none')

        # Apply importance sampling weights
        loss = (weights * loss_per_sample).mean()

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        return loss.item(), next_q_online, td_errors

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

    elif isinstance(net, RecursiveDQNNet):
        use_net = RecursiveDQNNet(
            net.nsize,
            net.n_states,
            net.n_actions,
            recursive_steps=net.recursive_steps
        )

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