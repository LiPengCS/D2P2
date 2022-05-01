import utils
from .experiment_utils import build_data
from model import LogisticRegression, TwoLayerNet, FiveLayerNet, EquiNNNorm
from trainer.baseline_trainer import BaselineTrainer
from trainer.baseline_trainer_sgd import BaselineTrainerSGD
import torch
import torch.nn as nn
from .experiment_utils import set_random_seed, load_data, makedir, grid_search, print_params, copy_result
from pipeline.baseline_pipeline import BaselinePipeline
from tqdm import tqdm
import time
import json
from utils import SummaryWriter
import shutil
import os

class BaselineExperiment(object):
    """Run baseline with one set of hyper parameters"""
    def __init__(self, data_dir, dataset, prep_space, method, model_name, tf_seed=1):
        self.data_dir = data_dir
        self.dataset = dataset
        self.prep_space = prep_space
        self.method = method
        self.tf_seed = tf_seed
        self.model_name = model_name

    def run(self, params):
        # set random seed
        set_random_seed(params)

        # load data
        X, y = load_data(self.data_dir, self.dataset, keepna=False)
        X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=params["seed"])

        # data prep pipeline
        prep_pipeline = BaselinePipeline(self.method, self.prep_space, self.tf_seed)
        X_train = prep_pipeline.fit_transform(X_train, X_val, X_test)
        X_val = prep_pipeline.transform(X_val)
        X_test = prep_pipeline.transform(X_test)

        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        X_test = torch.Tensor(X_test)

        # model
        input_dim = X_train.shape[1]
        output_dim = len(set(y.values.ravel()))
        
        set_random_seed(params)
        if self.model_name == "log":
            model = LogisticRegression(input_dim, output_dim)
        elif self.model_name == "two":
            model = TwoLayerNet(input_dim, output_dim)
        elif self.model_name == "five":
            model = FiveLayerNet(input_dim, output_dim)

        if self.method == "equinn_norm":
            model = EquiNNNorm(input_dim, self.prep_space, model)
            
        model = model.to(params["device"])

        # loss
        loss_fn = nn.CrossEntropyLoss()

        # optimizer
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["model_lr"],
            weight_decay=params["weight_decay"],
            momentum=params["momentum"]
        )
        # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=20, factor=0.1, threshold=0.001)
        model_scheduler = None

        if params["logging"]:
            logger = SummaryWriter()
            logger.add_baseline_pipeline(prep_pipeline.pipeline, global_step=0)
        else:
            logger = None

        baseline = BaselineTrainerSGD(model, loss_fn, model_optimizer, model_scheduler, params, writer=logger)
        result, model = baseline.fit(X_train, y_train, X_val, y_val, X_test, y_test)

        return result, model, logger

def run_baseline(data_dir, dataset, result_dir, prep_space, params, model_name, num_random=200):
    # baseline 1: default
    print(dataset, "default")
    result_dir_default = makedir([result_dir, "default"], remove_old_dir=True)
    tic = time.time()
    baseline_default = BaselineExperiment(data_dir, dataset, prep_space, "default", model_name)
    default_result = grid_search(baseline_default, params, result_dir_default)
    print("Finished in", time.time() - tic, "val acc:", default_result["best_val_acc"], "test acc", default_result["best_test_acc"])

    # baseline 2: random
    print(dataset, "random")
    best_val_acc = float("-inf")
    best_result = None
    best_seed = None
    result_dir_random_temp = makedir([result_dir, "random_temp"], remove_old_dir=True)
    result_dir_random_best = makedir([result_dir, "random_best"], remove_old_dir=True)

    for seed in tqdm(range(num_random)):
        params["random_tf_seed"] = seed
        baseline_random = BaselineExperiment(data_dir, dataset, prep_space, "random", model_name, tf_seed=seed)
        result = grid_search(baseline_random, params, result_dir_random_temp)

        if result["best_val_acc"] > best_val_acc:
            best_val_acc = result["best_val_acc"]
            best_seed = seed
            best_result = result
            copy_result(result_dir_random_temp, result_dir_random_best)

        if seed+1 in [10, 50, 100, 150, 200]:
            print(dataset, "random{}".format(seed+1))
            print("val acc:", best_result["best_val_acc"], "test acc", best_result["best_test_acc"])
            save_dir = makedir([result_dir, "random{}".format(seed + 1)], remove_old_dir=True)
            copy_result(result_dir_random_best, save_dir)

    if os.path.exists(result_dir_random_temp):
        shutil.rmtree(result_dir_random_temp)
    if os.path.exists(result_dir_random_best):
        shutil.rmtree(result_dir_random_best)

def run_equinn(data_dir, dataset, result_dir, prep_space, params, model_name):
    print(dataset, "equinn_norm")
    result_dir_equinn_norm = makedir([result_dir, "equinn_norm"], remove_old_dir=True)
    baseline_equinn_norm = BaselineExperiment(data_dir, dataset, prep_space, "equinn_norm", model_name)
    equinn_norm_result = grid_search(baseline_equinn_norm, params, result_dir_equinn_norm)
    print("Finished. val acc:", equinn_norm_result["best_val_acc"], "test acc", equinn_norm_result["best_test_acc"])

