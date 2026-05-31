import configparser
import pickle
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
    recursive_steps = config.getint("model_args", "recursive_steps", fallback=4)
    train_state = None

    if not checkpoint:
        env_args = get_env_args()
        print(f"Starting new model {name} | device: {device}")
        model = Model(env_args, name=name, device=device, l2_const=l2_const, recursive_steps=recursive_steps)
    else:
        state_path = checkpoint + "_state.pkl"
        model_path = checkpoint + ".pth.tar"

        if os.path.exists(model_path):
            print(f"Loaded model from {checkpoint} | device: {device}")
            model = Model.load(checkpoint + '.pth.tar', device=device)
        else:
            print(f"No model found at {model_path}, exiting")
            return

        if os.path.exists(state_path):
            print(f"Loaded training state from {state_path}")
            with open(state_path, "rb") as f:
                train_state = pickle.load(f)
        else:
            print(f"No training state found at {state_path}, starting with default training state")


    training_pipeline = TrainPipeline(model, save_dir, config, train_state=train_state)
    training_pipeline.run()


if __name__ == "__main__":
    conf_ini = "./config.ini"
    config = configparser.ConfigParser()
    config.read(conf_ini, encoding="utf-8")

    import multiprocessing

    manager = multiprocessing.Manager()
    init_endgame_database(manager)

    run_train_pipline(config)