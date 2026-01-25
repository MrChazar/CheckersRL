

import configparser
from deepdraughts.env import *
from deepdraughts.net_pytorch import Model
from deepdraughts.train_pipeline import TrainPipeline

def run_train_pipline(config):
    save_dir = "../savedata/"

    checkpoint = config.get("model_args", "checkpoint")
    name = config.get("model_args", "name")
    use_gpu = config.getboolean("model_args", "use_gpu")
    print(use_gpu)
    l2_const = config.getfloat("model_args", "l2_const")

    if not checkpoint:
        env_args = get_env_args()
        model = Model(env_args, name = name, use_gpu = use_gpu, l2_const = l2_const)
    else:
        model = Model.load(checkpoint)

    training_pipeline = TrainPipeline(model, save_dir, config)
    training_pipeline.run()

if __name__ == "__main__":
    conf_ini = "./config.ini"
    config = configparser.ConfigParser()
    config.read(conf_ini, encoding="utf-8")


    # train your own model
    import multiprocessing
    manager = multiprocessing.Manager()
    init_endgame_database(manager)
    run_train_pipline(config)