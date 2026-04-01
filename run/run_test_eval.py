import configparser
import os
from deepdraughts.env import *
from deepdraughts.net_pytorch import Model
from deepdraughts.train_pipeline import TrainPipeline
from deepdraughts.mcts_pure import MCTSPlayer as MCTS_pure

from deepdraughts.env import BLACK

# Fix for multiprocessing
try:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def evaluate_mcts(config):
    save_dir = "../savedata/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = config.get("model_args", "checkpoint", fallback=None)
    if checkpoint is None:
        print("!!! No checkpoint provided in config, exiting")
        return
    test_games = config.get("eval", "n_test_games", fallback=200)
    test_playout = config.get("eval", "n_test_playout", fallback=100)
    device = 'cpu'

    print(f"Loaded model from {checkpoint} | device: {device}")
    model = Model.load(checkpoint, device=device)

    training_pipeline = TrainPipeline(model, save_dir, config)

    mcts_player = MCTS_pure(c_puct=5, n_playout=test_playout)
    print('Starting test evaluation')
    training_pipeline.evaluate(test_playout, BLACK, mcts_player, test_games, model_name='mcts')


if __name__ == "__main__":
    conf_ini = "./config.ini"
    config = configparser.ConfigParser()
    config.read(conf_ini, encoding="utf-8")

    import multiprocessing

    manager = multiprocessing.Manager()
    # init_endgame_database(manager)

    evaluate_mcts(config)