from __future__ import print_function
import numpy as np
import random
from tensorboardX import SummaryWriter
import datetime
import os
import copy
import torch
from .dqn import ReplayBuffer, DQNAgent
from .game_collector import GameCollector, GAME_WIN, GAME_TIE, GAME_LOSS, PIECE_TAKEN
from .env import Game
from deepdraughts.mcts_pure import MCTSPlayer as MCTS_pure
from .env import *


class TrainPipeline():
    def __init__(self, model, dir_save, config, game_args=dict()):
        # Training Args
        self.max_epoch = config.getint("training_args", "max_epoch")
        self.batch_size = config.getint("training_args", "batch_size")  # Used for gradient update batch
        self.n_cores = config.getint("training_args", "n_cores")
        self.epochs = config.getint("training_args", "epochs")  # Gradient steps per epoch
        self.check_freq = config.getint("training_args", "check_freq")
        self.n_eval_games = config.getint("training_args", "n_eval_games")

        # DQN Args
        self.lr = config.getfloat("training_args", "learn_rate")
        self.gamma = config.getfloat("training_args", "gamma")
        self.buffer_size = config.getint("training_args", "buffer_size")
        self.eps_start = config.getfloat("training_args", "epsilon_start")
        self.eps_end = config.getfloat("training_args", "epsilon_end")
        self.eps_decay = config.getfloat("training_args", "epsilon_decay")
        self.target_update_freq = config.getint("training_args", "target_update_freq")
        self.n_steps = config.getint("model_args", "n_steps")

        self.game_args = game_args
        self.model = model
        self.dir_save = dir_save
        self.name = model.name

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.writer = SummaryWriter(self.dir_save + self.name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.n_epoch = 0
        self.global_step = 0
        self.current_eps = 1

    def get_epsilon(self):
        """Exponential decay for epsilon"""
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.global_step / self.eps_decay)

    def train(self, training_side=WHITE):
        # Collect Data (Self-Play)
        # collect 'n_cores' games per epoch loop to add variety
        self.current_eps = self.get_epsilon()
        new_transitions, _ = GameCollector.parallel_collect_selfplay(
            n_cores=self.n_cores,
            shared_model=self.model.policy_net,
            epsilon=self.current_eps,
            batch_size=self.batch_size, #self.n_cores * 2, # size of batch
            game_args=self.game_args,
            training_side=training_side,
            n_steps=self.n_steps,
            gamma=self.gamma
        )

        self.model.policy_net.to(device=self.model.device)

        # Add to Buffer
        for t in new_transitions:
            self.replay_buffer.push(*t)

        if len(self.replay_buffer) < self.batch_size:
            return 0  # Not enough data

        # Train Loop
        total_loss = 0
        q_max = -float('inf')
        q_mean = 0
        for _ in range(self.epochs):
            # Sample Batch
            #transitions = self.replay_buffer.sample(self.batch_size)
            transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.model.device).unsqueeze(1)
            # Transpose batch to (batch_board, batch_state, batch_action, ...)
            batch = list(zip(*transitions))

            # Stack numpy arrays
            b_board = np.stack(batch[0])
            b_state = np.stack(batch[1])
            b_action = np.array(batch[2])
            b_reward = np.array(batch[3])
            b_next_board = np.stack(batch[4])
            b_next_state = np.stack(batch[5])
            b_done = np.array(batch[6], dtype=np.uint8)

            batch_data = (b_board, b_state, b_action, b_reward, b_next_board, b_next_state, b_done)

            loss, q_vals, td_errors = self.model.train_step(batch_data, self.gamma, self.lr, weights, self.n_steps)
            td_errors_np = td_errors.detach().abs().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, td_errors_np)

            total_loss += loss
            q_max = max(q_max, q_vals.max().item())
            q_mean += q_vals.mean().item()
        q_mean /= self.epochs
        avg_loss = total_loss / self.epochs
        print(f"Epoch {self.n_epoch} | Buffer: {len(self.replay_buffer)} | Epsilon: {self.current_eps:.4f} | Avg loss: {avg_loss:.4f} | Q max: {q_max:.2f} | Q mean: {q_mean:.2f}")
        self.writer.add_scalar("loss", avg_loss, self.n_epoch)
        self.writer.add_scalar("epsilon", self.current_eps, self.n_epoch)

        return avg_loss

    def run(self):
        """Main Loop"""
        print("Starting DQN Training...")
        n_playout = 25
        mcts_side = BLACK
        for i in range(self.max_epoch):
            self.n_epoch += 1
            self.global_step += 1

            loss = self.train()

            # Sync Target Network
            tau = 0.006  # can adjust 0.001–0.01
            for target_param, online_param in zip(self.model.target_net.parameters(), self.model.policy_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            #if self.n_epoch % self.target_update_freq == 0:
            #    self.model.sync_target_network()

            # Checkpoint & Evaluate
            if self.n_epoch % self.check_freq == 0:
                print(f"Saving Checkpoint at epoch {self.n_epoch}", end=' - ')
                self.model.save(self.dir_save, self.n_epoch)

                mcts_player = MCTS_pure(c_puct=5, n_playout=n_playout)
                n_playout = self.evaluate(n_playout, mcts_side, mcts_player, 100, model_name='mcts')


    def evaluate(self, n_playout, opponent_side, opponent_player, evaluation_games=10, model_name='dqn'):
        agent = DQNAgent(self.model.policy_net, epsilon=0.0, device=self.model.device)
        wins = 0
        draws = 0
        losses = 0
        reward = 0
        for _ in range(evaluation_games):
            # evaluation game against MCTS
            game = Game(**self.game_args)
            while True:
                if game.current_player == opponent_side:
                    move, _ = opponent_player.get_action(game)
                    game_status = game.do_move(move)
                else:
                    move, _ = agent.get_action(game)
                    game_status = game.do_move(move)

                if game_is_over(game_status):
                    break
            # calculate reward
            pieces = game.current_board.get_pieces()
            mcts_pieces = len([x for x in pieces if x.player == opponent_side])
            dqn_pieces = len(pieces) - mcts_pieces
            reward += (12 - mcts_pieces) * PIECE_TAKEN
            reward -= (12 - dqn_pieces) * PIECE_TAKEN
            winner = game_winner(game_status)
            if winner == 0:
                draws += 1
                reward += GAME_TIE
            elif winner == opponent_side:
                losses += 1
                reward += GAME_LOSS
            else:
                wins += 1
                reward += GAME_WIN
        win_ratio = wins / evaluation_games
        loss_ratio = losses / evaluation_games
        avg_reward = reward / evaluation_games
        self.writer.add_scalar(f"{model_name} avg_reward", avg_reward, self.n_epoch)
        self.writer.add_scalar(f"{model_name} win_ratio", win_ratio, self.n_epoch)
        self.writer.add_scalar(f"{model_name} loss_ratio", loss_ratio, self.n_epoch)
        self.writer.add_scalar(f"{model_name} mcts_n_playout", n_playout, self.n_epoch)
        print(f'<{model_name}> Evaluation win ratio: {win_ratio:.2f} | Evaluation loss ratio: {loss_ratio:.2f} | Avg reward: {avg_reward:.4f} | mcts_n_playout: {n_playout}')
        if win_ratio > 0.75:
            n_playout *= 2
        return min(n_playout, 10000)
