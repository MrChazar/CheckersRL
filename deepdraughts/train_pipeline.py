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
        self.training_side = WHITE
        self.max_epoch = config.getint("training_args", "max_epoch")
        self.batch_size = config.getint("training_args", "batch_size")  # Used for gradient update batch
        self.n_cores = config.getint("training_args", "n_cores")
        self.epochs = config.getint("training_args", "epochs")  # Gradient steps per epoch


        # DQN Args
        self.lr = config.getfloat("training_args", "learn_rate")
        self.gamma = config.getfloat("training_args", "gamma", fallback=0.99)
        self.tau = config.getfloat("training_args", "tau", fallback=0.006)
        self.buffer_size = config.getint("training_args", "buffer_size", fallback=100_000)
        self.starting_buffer_games = config.getint("training_args", "starting_buffer_games", fallback=0)
        self.eps_start = config.getfloat("training_args", "epsilon_start")
        self.eps_end = config.getfloat("training_args", "epsilon_end")
        self.eps_decay = config.getfloat("training_args", "epsilon_decay")
        self.beta_start = config.getfloat("training_args", "beta_start", fallback=0.4)
        self.beta_frames = config.getfloat("training_args", "beta_frames", fallback=200_000)
        self.n_steps = config.getint("model_args", "n_steps")

        # Evaluation Args
        self.check_freq = config.getint("evaluation", "eval_freq", fallback=100)
        self.n_eval_games = config.getint("evaluation", "n_eval_games", fallback=100)

        self.game_args = game_args
        self.model = model
        self.dir_save = dir_save
        self.name = model.name

        self.replay_buffer = ReplayBuffer(self.buffer_size, device='cpu')
        self.writer = SummaryWriter(self.dir_save + self.name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.n_epoch = 0
        self.global_step = 0
        self.current_eps = self.get_epsilon()
        self.current_beta = self.get_beta()

        # Load starting transitions
        if self.starting_buffer_games > 0:
            transitions = self.get_transitions(self.starting_buffer_games)
            print(f'Collected {self.starting_buffer_games} games, {len(transitions)} transitions, adding to buffer...')
            self.replay_buffer.push_multiple(transitions)

    def get_epsilon(self):
        """Exponential decay for epsilon"""
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.global_step / self.eps_decay)

    def get_beta(self):
        return min(1.0, self.beta_start + self.global_step * (1.0 - self.beta_start) / self.beta_frames)

    def get_transitions(self, n_games = None):
        if n_games is None:
            n_games = self.n_cores * 4

        new_transitions, _ = GameCollector.parallel_collect_selfplay(
            n_cores=self.n_cores,
            shared_model=self.model.policy_net,
            epsilon=self.current_eps,
            batch_size = n_games,  # size of batch
            game_args=self.game_args,
            training_side=self.training_side,
            n_steps=self.n_steps,
            gamma=self.gamma,
            device=self.model.device,
        )

        return new_transitions

    def train(self, training_side=WHITE):
        self.training_side = training_side
        # Collect Data (Self-Play)
        # collect 'n_cores' games per epoch loop to add variety
        self.current_eps = self.get_epsilon()
        self.current_beta = self.get_beta()

        # self.model.policy_net.to(device=self.model.device)

        # Add to Buffer
        new_transitions = self.get_transitions()
        self.replay_buffer.push_multiple(new_transitions)

        if len(self.replay_buffer) < self.batch_size:
            return 0  # Not enough data

        # Train Loop
        total_loss = 0
        q_max = -float('inf')
        q_mean = 0
        for _ in range(self.epochs):
            # Sample Batch
            #transitions = self.replay_buffer.sample(self.batch_size)
            transitions, indices, weights = self.replay_buffer.sample(self.batch_size, self.current_beta)
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
        print(
            f"Epoch {self.n_epoch} | Buffer: {self.replay_buffer.pos} | Epsilon: {self.current_eps:.4f} | Avg loss: {(avg_loss * 10):.4f} | Q max: {q_max:.2f} | Q mean: {q_mean:.2f} | Beta: {self.current_beta:.4f}")
        self.writer.add_scalar("loss", avg_loss, self.n_epoch)
        self.writer.add_scalar("epsilon", self.current_eps, self.n_epoch)
        self.writer.add_scalar("beta", self.current_beta, self.n_epoch)
        self.writer.add_scalar("q_mean", q_mean, self.n_epoch)
        self.writer.add_scalar("q_max", q_max, self.n_epoch)

        return avg_loss

    def run(self):
        """Main Loop"""
        print("Starting DQN Training...")
        n_playout = 100
        mcts_side = BLACK
        self.model.policy_net.to(device=self.model.device)

        # mcts_player = MCTS_pure(c_puct=5, n_playout=n_playout)
        # n_playout = self.evaluate(n_playout, mcts_side, mcts_player, 100, model_name='mcts')

        for i in range(self.max_epoch):
            self.n_epoch += 1
            self.global_step += 1

            loss = self.train()

            # Sync Target Network
            tau = self.tau  # can adjust 0.001–0.01
            for target_param, online_param in zip(self.model.target_net.parameters(),
                                                  self.model.policy_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            # if self.n_epoch % self.target_update_freq == 0:
            #    self.model.sync_target_network()

            # Checkpoint & Evaluate
            if self.n_epoch % self.check_freq == 0:
                print(f"Saving Checkpoint at epoch {self.n_epoch}", end=' - ')
                self.model.save(self.dir_save, self.n_epoch)

                mcts_player = MCTS_pure(c_puct=5, n_playout=n_playout)
                n_playout = self.evaluate(n_playout, mcts_side, mcts_player, self.n_eval_games, model_name='mcts')

    def evaluate(self, n_playout, opponent_side, opponent_player, evaluation_games=10, model_name='dqn'):
        net = self.model.policy_net
        agent = DQNAgent(net, epsilon=0.0, device=self.model.device)
        wins = 0
        draws = 0
        losses = 0
        reward = 0
        for game in range(evaluation_games):
            print(f'Evaluating game {game}/{evaluation_games}')
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

        # reset device
        net.to(device=self.model.device)

        self.writer.add_scalar(f"{model_name} avg_reward", avg_reward, self.n_epoch)
        self.writer.add_scalar(f"{model_name} win_ratio", win_ratio, self.n_epoch)
        self.writer.add_scalar(f"{model_name} loss_ratio", loss_ratio, self.n_epoch)
        self.writer.add_scalar(f"{model_name} mcts_n_playout", n_playout, self.n_epoch)
        print(
            f'<{model_name}> Evaluation win ratio: {win_ratio:.2f} | Evaluation loss ratio: {loss_ratio:.2f} | Avg reward: {avg_reward:.4f} | mcts_n_playout: {n_playout}')
        if win_ratio > 0.75:
            n_playout *= 2
        return min(n_playout, 10000)
