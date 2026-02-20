from __future__ import print_function
import numpy as np
import random
from tensorboardX import SummaryWriter
import datetime
import os
import torch
from .dqn import ReplayBuffer
from .game_collector import GameCollector
from .env import Game


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

        self.game_args = game_args
        self.model = model
        self.dir_save = dir_save
        self.name = model.name

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.writer = SummaryWriter(self.dir_save + self.name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.n_epoch = 0
        self.global_step = 0

    def get_epsilon(self):
        """Exponential decay for epsilon"""
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.global_step / self.eps_decay)

    def train(self):
        # Collect Data (Self-Play)
        # collect 'n_cores' games per epoch loop to add variety
        current_eps = self.get_epsilon()
        new_transitions, _ = GameCollector.parallel_collect_selfplay(
            n_cores=self.n_cores,
            shared_model=self.model.policy_net,
            epsilon=current_eps,
            batch_size=self.n_cores * 2, # size of batch
            game_args=self.game_args
        )

        if self.model.use_gpu:
            self.model.policy_net.cuda()

        # Add to Buffer
        for t in new_transitions:
            self.replay_buffer.push(*t)

        print(f"Epoch {self.n_epoch} | Buffer: {len(self.replay_buffer)} | Epsilon: {current_eps:.4f}")

        if len(self.replay_buffer) < self.batch_size:
            return 0  # Not enough data

        # Train Loop
        total_loss = 0
        for _ in range(self.epochs):
            # Sample Batch
            transitions = self.replay_buffer.sample(self.batch_size)
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

            loss = self.model.train_step(batch_data, self.gamma, self.lr)
            total_loss += loss

        avg_loss = total_loss / self.epochs
        self.writer.add_scalar("loss", avg_loss, self.n_epoch)
        self.writer.add_scalar("epsilon", current_eps, self.n_epoch)

        return avg_loss

    def run(self):
        """Main Loop"""
        print("Starting DQN Training...")
        for i in range(self.max_epoch):
            self.n_epoch += 1
            self.global_step += 1

            loss = self.train()

            # Sync Target Network
            if self.n_epoch % self.target_update_freq == 0:
                self.model.sync_target_network()

            # Checkpoint & Evaluate
            if self.n_epoch % self.check_freq == 0:
                print(f"Saving Checkpoint at epoch {self.n_epoch}")
                self.model.save(self.dir_save, self.n_epoch)

                # Optional: Maybe add win_rate ?