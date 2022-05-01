import utils
import argparse
import time
from prep_space import space
from experiment.baseline_experiment import run_baseline
from experiment.auto_prep_experiment import run_auto_prep
# from experiment.deepnn_experiment import run_deepnn
# from experiment.baseline_experiment import run_equinn
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="obesity")
parser.add_argument('--data_dir', default="data/data10")
parser.add_argument('--result_dir', default="result")
parser.add_argument('--model', default="two", choices=["log", "two"])
parser.add_argument('--baseline', action="store_true", default=False)
# parser.add_argument('--space', default="2", choices=["1", "2", "3", "repeat2", "repeat1"])
parser.add_argument('--diff_method', default="num_diff", choices=["num_diff", "relax", "sf", "st"])
parser.add_argument('--sample', action="store_true", default=False)
parser.add_argument('--first_order', action="store_true", default=False)
parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--split_seed', default=1, type=int)
parser.add_argument('--test', action="store_true", default=False)
parser.add_argument('--no_crash', action="store_true", default=False)
parser.add_argument('--order', action="store_true", default=False)
parser.add_argument('--order_feature', action="store_true", default=False)
# parser.add_argument('--pre_norm', action="store_true", default=False)
# parser.add_argument('--equinn', action="store_true", default=False)
# parser.add_argument('--deepnn', action="store_true", default=False)
args = parser.parse_args()

# define hyper parameters
params = {
    "num_epochs": 2000,
    # "num_epochs": 500,
    "batch_size": 128,
    "device": "cpu",
    "model_lr": [0.01, 0.1, 1],
    "weight_decay": 0,
    "split_seed": args.split_seed,
    "seed": args.seed,
    "init_method": "default",
    "save_model": True,
    "momentum": 0.9,
    "logging": True,
    "no_crash": args.no_crash,
    "order": args.order,
    "order_feature": args.order_feature,
    "pre_norm": True
}

if args.gpu and torch.cuda.is_available():
    params["device"] = "cuda"

auto_prep_params = {
    "prep_lr": [0.01, 0.1, 1],
    "temperature": 0.1,
    "grad_clip": None,
    "diff_method": args.diff_method,
    "sample": args.sample,
    "first_order_only": args.first_order,
}

if args.test:
    params["num_epochs"] = 1
    params["model_lr"] = [1]
    auto_prep_params["prep_lr"] = [0.1]
    
args.result_dir = utils.makedir([args.result_dir, args.dataset, "seed_{}".format(args.seed)])

tic = time.time()
if args.baseline:
    result_dir = utils.makedir([args.result_dir, "baseline"])
    run_baseline(args.data_dir, args.dataset, result_dir, space, params, args.model)
    utils.save_prep_space(space, result_dir)
else:
    params.update(auto_prep_params)
    if args.order:
        result_dir = utils.makedir([args.result_dir, "auto_prep_order"])
    elif args.order_feature:
        result_dir = utils.makedir([args.result_dir, "auto_prep_order_feature"])
    else:
        result_dir = utils.makedir([args.result_dir, "auto_prep"])
    run_auto_prep(args.data_dir, args.dataset, result_dir, space, params, args.model)
    utils.save_prep_space(space, result_dir)

print(time.time() - tic)