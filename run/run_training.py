import configparser
import os
import torch
from deepdraughts.env import *
from deepdraughts.ppo_net import PPOModel
from deepdraughts.train_pipeline import TrainPipeline


def run_train_pipeline(config):
    save_dir = "../savedata/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = config.get("model_args", "checkpoint", fallback=None)
    name = config.get("model_args", "name", fallback="PPO_Checkers")

    requested_device = config.get("model_args", "device", fallback="cuda")
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"

    lr = config.getfloat("training_args", "learn_rate", fallback=3e-4)

    if not checkpoint or not os.path.exists(checkpoint):
        env_args = get_env_args()
        print(f"Starting new model {name} | device: {device}")
        model = PPOModel(env_args, name=name, device=device, lr=lr)
    else:
        print(f"Loaded model from {checkpoint} | device: {device}")
        model = PPOModel.load(checkpoint, device)

    training_pipeline = TrainPipeline(model, save_dir, config)
    training_pipeline.run()


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp

        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config = configparser.ConfigParser()
    config.read("./config.ini", encoding="utf-8")

    import multiprocessing

    manager = multiprocessing.Manager()

    init_endgame_database(manager)

    run_train_pipeline(config)