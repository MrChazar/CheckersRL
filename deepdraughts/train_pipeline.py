import numpy as np
import torch
from tensorboardX import SummaryWriter
import datetime
from .ppo import RolloutBuffer, PPOAgent
from .game_collector import GameCollector, GAME_WIN, GAME_TIE, GAME_LOSS, PIECE_TAKEN
from .env import Game, game_is_over, game_winner, WHITE, BLACK
from deepdraughts.mcts_pure import MCTSPlayer as MCTS_pure


class TrainPipeline():
    def __init__(self, model, dir_save, config, game_args=dict()):
        self.max_epoch = config.getint("training_args", "max_epoch")
        self.batch_size = config.getint("training_args", "batch_size")
        self.n_cores = config.getint("training_args", "n_cores")
        self.epochs = config.getint("training_args", "epochs")  # PPO Epochs
        self.check_freq = config.getint("training_args", "check_freq")
        self.game_args = game_args
        self.model = model
        self.dir_save = dir_save
        self.name = model.name

        self.rollout_buffer = RolloutBuffer()
        self.writer = SummaryWriter(self.dir_save + self.name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.n_epoch = 0

    def train(self, training_side=WHITE):
        # collecting data for rollouts
        raw_rollouts, _ = GameCollector.parallel_collect_selfplay(
            n_cores=self.n_cores,
            shared_model=self.model.policy_net,
            batch_size=self.n_cores * 2,
            game_args=self.game_args,
            training_side=training_side
        )

        self.model.policy_net.to(device=self.model.device)

        # mapping data
        self.rollout_buffer.clear()
        self.rollout_buffer.boards = raw_rollouts['b']
        self.rollout_buffer.states = raw_rollouts['s']
        self.rollout_buffer.actions = raw_rollouts['a']
        self.rollout_buffer.logprobs = raw_rollouts['lp']
        self.rollout_buffer.rewards = raw_rollouts['r']
        self.rollout_buffer.values = raw_rollouts['v']
        self.rollout_buffer.dones = raw_rollouts['d']
        self.rollout_buffer.masks = raw_rollouts['m']
        self.rollout_buffer.advantages = raw_rollouts['adv']
        self.rollout_buffer.returns = raw_rollouts['ret']

        if len(self.rollout_buffer.boards) < self.batch_size:
            return 0

        avg_loss, p_loss, v_loss = self.model.update(self.rollout_buffer, self.epochs, self.batch_size)

        print(
            f"Epoch {self.n_epoch} | Rollout size: {len(self.rollout_buffer.boards)} | Loss: {avg_loss:.4f} | Actor: {p_loss:.4f} | Critic: {v_loss:.4f}")
        self.writer.add_scalar("loss/total", avg_loss, self.n_epoch)
        self.writer.add_scalar("loss/policy", p_loss, self.n_epoch)
        self.writer.add_scalar("loss/value", v_loss, self.n_epoch)

        return avg_loss

    def run(self):
        print("Starting PPO Training...")
        n_playout = 25
        mcts_side = BLACK
        for i in range(self.max_epoch):
            self.n_epoch += 1
            self.train()

            if self.n_epoch % self.check_freq == 0:
                print(f"Saving Checkpoint at epoch {self.n_epoch}")
                self.model.save(self.dir_save, self.n_epoch)

                # evaluation against mcts
                mcts_player = MCTS_pure(c_puct=5, n_playout=n_playout)
                n_playout = self.evaluate(n_playout, mcts_side, mcts_player, evaluation_games=20, model_name='mcts')

    def evaluate(self, n_playout, opponent_side, opponent_player, evaluation_games=10, model_name='ppo'):
        agent = PPOAgent(self.model.policy_net, device=self.model.device)
        wins, losses, draws = 0, 0, 0

        for _ in range(evaluation_games):
            game = Game(**self.game_args)
            while True:
                if game.current_player == opponent_side:
                    move, _ = opponent_player.get_action(game)
                else:
                    move, _, _, _ = agent.get_action(game, deterministic=True)

                game_status = game.do_move(move)
                if game_is_over(game_status):
                    break

            winner = game_winner(game_status)

            if winner == opponent_side:
                losses += 1
            elif winner == 0:
                draws += 1
            else:
                wins += 1

        win_ratio = wins / evaluation_games
        loss_ratio = losses / evaluation_games
        draw_ratio = draws / evaluation_games

        print(
            f'<{model_name}> Evaluation against MCTS ({n_playout} playouts) - Win: {win_ratio:.2f} | Draw: {draw_ratio:.2f} | Loss: {loss_ratio:.2f}')

        self.writer.add_scalar("Eval_vs_MCTS/Win_Rate", win_ratio, self.n_epoch)
        self.writer.add_scalar("Eval_vs_MCTS/Draw_Rate", draw_ratio, self.n_epoch)
        self.writer.add_scalar("Eval_vs_MCTS/Loss_Rate", loss_ratio, self.n_epoch)
        self.writer.add_scalar("Eval_vs_MCTS/MCTS_Playouts_Difficulty", n_playout, self.n_epoch)

        if win_ratio > 0.75:
            n_playout *= 2

        return min(n_playout, 10000)