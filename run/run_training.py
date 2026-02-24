import configparser
import os
from deepdraughts.env import *
from deepdraughts.net_pytorch import Model
from deepdraughts.train_pipeline import TrainPipeline

# Fix for multiprocessing
try:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def run_train_pipline(config):
    save_dir = "../savedata/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = config.get("model_args", "checkpoint", fallback=None)
    name = config.get("model_args", "name", fallback="DQN")
    device = config.get("model_args", "device", fallback="cuda")
    l2_const = config.getfloat("model_args", "l2_const", fallback=0)

    if not checkpoint:
        env_args = get_env_args()
        print(f"Starting new model {name} | device: {device}")
        model = Model(env_args, name=name, device=device, l2_const=l2_const)
    else:
        print(f"Loaded model from {checkpoint} | device: {device}")
        model = Model.load(checkpoint)

    training_pipeline = TrainPipeline(model, save_dir, config)
    training_pipeline.run()


if __name__ == "__main__":
    conf_ini = "./config.ini"
    config = configparser.ConfigParser()
    config.read(conf_ini, encoding="utf-8")

    import multiprocessing

    manager = multiprocessing.Manager()
    init_endgame_database(manager)

    run_train_pipline(config)