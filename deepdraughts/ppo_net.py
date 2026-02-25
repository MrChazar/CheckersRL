import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import datetime


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ActorCriticNet(nn.Module):
    def __init__(self, nsize, n_states, n_actions):
        super(ActorCriticNet, self).__init__()
        self.nsize = nsize
        self.n_states = n_states
        self.n_actions = n_actions

        # Feature extractor
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.st_fc1 = nn.Linear(n_states, 64)
        self.fc_common = nn.Linear(128 * nsize * nsize + 64, 512)

        # Actor Head (Policy)
        self.actor_fc = nn.Linear(512, 256)
        self.actor_head = nn.Linear(256, n_actions)

        # Critic Head (Value)
        self.critic_fc = nn.Linear(512, 256)
        self.critic_head = nn.Linear(256, 1)

        _initialize_weights(self)

    def forward(self, vec_board, vec_state):
        if len(vec_board.shape) == 3:
            vec_board = torch.unsqueeze(vec_board, 0)
            vec_state = torch.unsqueeze(vec_state, 0)

        x = F.relu(self.bn1(self.conv1(vec_board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        y = F.relu(self.st_fc1(vec_state))
        combined = torch.cat((x, y), 1)
        feat = F.relu(self.fc_common(combined))

        # Actor
        a_out = F.relu(self.actor_fc(feat))
        logits = self.actor_head(a_out)

        # Critic
        c_out = F.relu(self.critic_fc(feat))
        value = self.critic_head(c_out)

        return logits, value


class PPOModel():
    def __init__(self, env_args, name="ppo_model", device='cpu', lr=3e-4):
        nsize, _, n_states, n_actions = env_args
        self.nsize = nsize
        self.n_states = n_states
        self.n_actions = n_actions
        self.name = name
        self.device = device
        self.lr = lr

        self.policy_net = ActorCriticNet(nsize, n_states, n_actions).to(device=self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-5)

        self.clip_ratio = 0.2
        self.c_value = 0.5
        self.c_entropy = 0.01


    def evaluate_actions(self, b_board, b_state, b_action, b_mask):
        logits, values = self.policy_net(b_board, b_state)

        logits[~b_mask] = -1e9

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logprobs = dist.log_prob(b_action.squeeze(-1))
        entropy = dist.entropy()

        return logprobs, values.squeeze(-1), entropy

    def update(self, rollouts, epochs, batch_size):
        b_board = torch.tensor(np.stack(rollouts.boards), dtype=torch.float32).to(self.device)
        b_state = torch.tensor(np.stack(rollouts.states), dtype=torch.float32).to(self.device)
        b_action = torch.tensor(rollouts.actions, dtype=torch.long).to(self.device).unsqueeze(-1)
        b_old_logprobs = torch.tensor(rollouts.logprobs, dtype=torch.float32).to(self.device)
        b_returns = torch.tensor(rollouts.rewards, dtype=torch.float32).to(self.device)
        b_advantages = torch.tensor(rollouts.values, dtype=torch.float32).to(self.device)
        b_masks = torch.tensor(np.stack(rollouts.masks), dtype=torch.bool).to(self.device)

        # for training stabilisation
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        dataset_size = len(b_board)
        indices = np.arange(dataset_size)

        avg_loss, avg_vloss, avg_ploss = 0, 0, 0
        steps = 0

        self.policy_net.train()
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                logprobs, values, entropy = self.evaluate_actions(
                    b_board[idx], b_state[idx], b_action[idx], b_masks[idx]
                )

                ratios = torch.exp(logprobs - b_old_logprobs[idx])

                surr1 = ratios * b_advantages[idx]
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values, b_returns[idx])
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()

                avg_loss += loss.item()
                avg_vloss += critic_loss.item()
                avg_ploss += actor_loss.item()
                steps += 1

        return avg_loss / steps, avg_ploss / steps, avg_vloss / steps

    def save(self, checkpoint_dir, epoch, is_best=False):
        now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        name = self.name + "_best" if is_best else self.name
        savepath = os.path.join(checkpoint_dir, '{}_epoch{}_{}.pth.tar'.format(name, epoch, now_time))
        torch.save({
            'nsize': self.nsize, 'n_states': self.n_states, 'n_actions': self.n_actions,
            'name': self.name, 'n_epoch': epoch, 'device': self.device,
            'model': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, savepath)

    @classmethod
    def load(cls, model_file, device='cpu'):
        model_params = torch.load(model_file, map_location=torch.device(device=device))
        env_args = (model_params['nsize'], None, model_params['n_states'], model_params['n_actions'])
        model = cls(env_args, name=model_params.get('name', 'ppo_loaded'), device=device)
        model.policy_net.load_state_dict(model_params['model'])
        model.optimizer.load_state_dict(model_params['optimizer'])
        return model