import configparser
import os
import torch
from deepdraughts.env import *
from deepdraughts.mcts_pure import MCTSPlayer as MCTS_pure
from deepdraughts.ppo import PPOAgent
from deepdraughts.gui import GUI
from deepdraughts.mcts_alphazero import MCTSPlayer_alphazero as MCTS_alphazero
from deepdraughts.ppo_net import PPOModel


class PPO_GUI_Wrapper:
    def __init__(self, ppo_agent):
        self.agent = ppo_agent

    def get_action(self, game):
        move, log_prob, value, mask = self.agent.get_action(game, deterministic=True)
        return move, value


def run_human_play(config):
    play_with = config.get("playing_args", "play_with")
    play_using_white = config.getboolean("playing_args", "play_using_white")
    using_endgame_database = config.getboolean("playing_args", "using_endgame_database")

    if using_endgame_database:
        init_endgame_database(None)

    if play_with == "human":
        GUI().run()
    elif play_with == "alphazero":
        # Removed dqn and alphazero but for final solution we must fit every model in this file
        pass
    elif play_with == "ppo":
        checkpoint = config.get("model_args", "checkpoint")

        requested_device = config.get("model_args", "device", fallback="cpu")
        device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"

        play_with_PPO(play_using_white, checkpoint, device)
    else:
        n_playout = config.getint("playing_args", "n_playout")
        mcts_player = MCTS_pure(c_puct=5, n_playout=n_playout)
        gui = GUI()
        if play_using_white:
            gui.run(player_black=AI_PLAYER, policy_black=mcts_player)
        else:
            gui.run(player_white=AI_PLAYER, policy_white=mcts_player)


def play_with_PPO(play_using_white=True, checkpoint=None, device='cpu'):
    gui = GUI()
    if checkpoint and os.path.exists(checkpoint):
        print(f"PPO loaded from:: {checkpoint} | device: {device}")
        model_container = PPOModel.load(checkpoint, device)
    else:
        print(f"Initialization of random net | device: {device}")
        model_container = PPOModel(get_env_args(), device=device)

    model_container.policy_net.eval()

    ppo_player = PPOAgent(model_net=model_container.policy_net, device=device,
                          side=BLACK if play_using_white else WHITE)
    gui_player = PPO_GUI_Wrapper(ppo_player)

    print(f"PPO ready to play {'black' if play_using_white else 'white'}")

    if play_using_white:
        gui.run(player_black=AI_PLAYER, policy_black=gui_player)
    else:
        gui.run(player_white=AI_PLAYER, policy_white=gui_player)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini", encoding="utf-8")
    run_human_play(config)