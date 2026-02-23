from deepdraughts.env import *
from deepdraughts.mcts_pure import MCTSPlayer as MCTS_pure
from deepdraughts.dqn import DQNAgent
from deepdraughts.gui import GUI
from deepdraughts.mcts_alphazero import MCTSPlayer_alphazero as MCTS_alphazero
from deepdraughts.net_pytorch import Model
import configparser
import torch
import os


def run_human_play(config):
    play_with = config.get("playing_args", "play_with")
    play_using_white = config.getboolean("playing_args", "play_using_white")
    using_endgame_database = config.getboolean("playing_args", "using_endgame_database")

    if using_endgame_database:
        print("Using endgame database")
        init_endgame_database(None)

    if play_with == "human":
        play_with_human()

    elif play_with == "alphazero":
        # we download n_playout only here
        n_playout = config.getint("playing_args", "n_playout")
        checkpoint = config.get("model_args", "checkpoint")
        name = config.get("model_args", "name")
        use_gpu = config.getboolean("model_args", "use_gpu")
        l2_const = config.getfloat("model_args", "l2_const")
        play_with_alphazero(play_using_white, n_playout, checkpoint, name, use_gpu, l2_const)

    elif play_with == "dqn":
        checkpoint = config.get("model_args", "checkpoint")
        device = config.get("model_args", "device")
        play_with_DQN(play_using_white, checkpoint, device)

    else:
        n_playout = config.getint("playing_args", "n_playout")
        play_with_pureMCTS(play_using_white, n_playout)


def play_with_human():
    gui = GUI()
    gui.run()


def play_with_DQN(play_using_white=True, checkpoint=None, device='cpu'):
    gui = GUI()

    # We load model
    if checkpoint and os.path.exists(checkpoint):
        print(f"DQN loaded from:: {checkpoint}")
        #try:
        model_container = Model.load(checkpoint, device)
        policy_net = model_container.policy_net
        #except Exception as e:
        #    print(f"Critical error during import: {e}")
        #    return
    else:
        print("Initialization of random net")
        env_args = get_env_args()
        # We create new model
        model_container = Model(env_args, device=device)
        policy_net = model_container.policy_net

    # setting up net in eval mode
    policy_net.eval()

    # creation of player
    dqn_player = DQNAgent(model_net=policy_net, epsilon=0.0, device=device, side=BLACK if play_using_white else WHITE)

    print(f"DQN ready to play {'black' if play_using_white else 'white'}")

    # setting up GUI
    if play_using_white:
        gui.run(player_black=AI_PLAYER, policy_black=dqn_player)
    else:
        gui.run(player_white=AI_PLAYER, policy_white=dqn_player)


def play_with_alphazero(play_using_white=True, n_playout=1000, checkpoint="",
                        name="default", use_gpu=False, l2_const=1e-4):
    gui = GUI()
    if not checkpoint:
        env_args = get_env_args()
        model = Model(env_args, name=name, use_gpu=use_gpu, l2_const=l2_const)
    else:
        model = Model.load(checkpoint)

    mcts_player = MCTS_alphazero(model.policy_value_fn, c_puct=5, n_playout=n_playout, selfplay=False)
    if play_using_white:
        gui.run(player_black=AI_PLAYER, policy_black=mcts_player)
    else:
        gui.run(player_white=AI_PLAYER, policy_white=mcts_player)


def play_with_pureMCTS(play_using_white=True, n_playout=1600):
    gui = GUI()
    mcts_player = MCTS_pure(c_puct=5, n_playout=n_playout)
    if play_using_white:
        gui.run(player_black=AI_PLAYER, policy_black=mcts_player)
    else:
        gui.run(player_white=AI_PLAYER, policy_white=mcts_player)


if __name__ == "__main__":
    conf_ini = "./config.ini"
    config = configparser.ConfigParser()
    config.read(conf_ini, encoding="utf-8")

    run_human_play(config)