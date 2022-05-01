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
from copy import deepcopy
from itertools import permutations

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="obesity")
parser.add_argument('--data_dir', default="data/data10")
parser.add_argument('--result_dir', default="all_order_random")
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
# parser.add_argument('--equinn', action="store_true", default=False)
# parser.add_argument('--deepnn', action="store_true", default=False)
args = parser.parse_args()

n_steps = len(space)
all_order_idx = list(permutations(range(1, n_steps)))
all_spaces = []

for order_idx in all_order_idx:
    space_name = "_".join([str(i) for i in order_idx])
    print(space_name)

    skip_dir = os.path.join(args.result_dir, args.dataset, "seed_{}".format(args.seed), "space_{}".format(space_name))
    if os.path.exists(skip_dir):
        print("skip", space_name)
        continue

    new_space = [deepcopy(space[0])]
    for i in order_idx:
        new_space.append(deepcopy(space[i]))

    # define hyper parameters
    params = {
        "num_epochs": 2000,
        # "num_epochs": 500,
        "batch_size": 128,
        "device": "cpu",
        "model_lr": [1, 0.1, 0.01],
        "weight_decay": 0,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "init_method": "default",
        "save_model": True,
        "momentum": 0.9,
        "logging": True
    }

    if args.gpu and torch.cuda.is_available():
        params["device"] = "cuda"

    auto_prep_params = {
        "prep_lr": [1, 0.1, 0.01],
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

    result_dir = utils.makedir([args.result_dir, args.dataset, "seed_{}".format(args.seed), "space_{}".format(space_name)], remove_old_dir=True)

    tic = time.time()
    if args.baseline:
        result_dir = utils.makedir([result_dir, "baseline"])
        run_baseline(args.data_dir, args.dataset, result_dir, new_space, params, args.model, num_random=100)
        utils.save_prep_space(new_space, result_dir)
    else:
        params.update(auto_prep_params)
        result_dir = utils.makedir([result_dir, "auto_prep"])
        run_auto_prep(args.data_dir, args.dataset, result_dir, new_space, params, args.model)
        utils.save_prep_space(new_space, result_dir)

    print(time.time() - tic)