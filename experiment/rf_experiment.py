import utils
from .experiment_utils import build_data
from trainer.baseline_trainer import BaselineTrainer
from trainer.baseline_trainer_sgd import BaselineTrainerSGD
from pipeline.auto_prep_pipeline import AutoPrepPipeline
import torch
import torch.nn as nn
from .experiment_utils import set_random_seed, load_data, makedir, grid_search, print_params
from pipeline.baseline_pipeline import BaselinePipeline
from .baseline_experiment import BaselineExperiment
from tqdm import tqdm
import time

class NonNNExperiment(object):
    def __init__(self, data_dir, dataset, auto_prep_space, transfer_prep_path, baseline_prep_space, method, model):
        self.data_dir = data_dir
        self.dataset = dataset
        self.auto_prep_space = auto_prep_space
        self.transfer_prep_path = transfer_prep_path
        self.baseline_prep_space = baseline_prep_space
        self.method = method
        self.model = model


    def run(self, params, log_dir):
        # set random seed
        set_random_seed(params)

        # load data
        X, y = load_data(self.data_dir, self.dataset, keepna=False)
        X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=params["seed"])

        # data prep pipeline
        prep_pipeline = AutoPrepPipeline(self.prep_space, temperature=params["temperature"],
                                 use_sample=params["sample"], diff_method=params["diff_method"])
        prep_pipeline.init_parameters(X_train, X_val, X_test)
        # print(self.transfer_prep_path)
        prep_pipeline.load_state_dict(torch.load(self.transfer_prep_path))

        for alpha in prep_pipeline.parameters():
            alpha.requires_grad = False

        X_train = prep_pipeline.fit(X_train)
        X_val = prep_pipeline.transform(X_val, X_type="val")
        X_test = prep_pipeline.transform(X_test, X_type="test")

        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        X_test = torch.Tensor(X_test)

        # model
        input_dim = X_train.shape[1]
        output_dim = len(set(y.values.ravel()))

        model = FiveLayerNet(input_dim, output_dim)    
        model = model.to(params["device"])

        # loss
        loss_fn = nn.CrossEntropyLoss()

        if params["sgd"]:
            patience = 20
        else:
            patience = 100
            
        # optimizer
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["model_lr"],
            weight_decay=params["weight_decay"]
        )
        model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=100, factor=0.1, threshold=0.001)

        if params["sgd"]:
            baseline = BaselineTrainerSGD(model, loss_fn, model_optimizer, model_scheduler, params, log_dir=log_dir)
        else:
            baseline = BaselineTrainer(model, loss_fn, model_optimizer, model_scheduler, params, log_dir=log_dir)
            
        result = baseline.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        return result, None

def run_nonnn(data_dir, dataset, result_dir, auto_prep_space, transfer_prep_path, baseline_prep_space, params, num_random=10):
    print(dataset, "nonnn", "default")
    result_dir_default = makedir([result_dir, "default"], remove_old_dir=True)
    tic = time.time()
    baseline_default = NonNNExperiment(data_dir, dataset, None, None, baseline_prep_space, "default", "rf")
    default_result = grid_search(baseline_default, params, result_dir_default)
    print("Finished in", time.time() - tic, "val acc:", default_result["best_val_acc"], "test acc", default_result["best_test_acc"])
    
    # baseline 2: random
    print(dataset, "random")
    best_val_acc = float("-inf")
    best_result = None
    best_seed = None
    
    for seed in tqdm(range(num_random)):
        baseline_random = NonNNExperiment(data_dir, dataset, None, None, baseline_prep_space, "random", "rf", tf_seed=seed)
        result = grid_search(baseline_random, params, None)
    
        if result["best_val_acc"] > best_val_acc:
            best_val_acc = result["best_val_acc"]
            best_seed = seed
            best_result = result
    
    print(dataset, "random 10")
    result_dir_random = makedir([result_dir, "random10"], remove_old_dir=True)
    baseline_random = NonNNExperiment(data_dir, dataset, baseline_prep_space, "random", "rf", tf_seed=best_seed)
    random_result = grid_search(baseline_random, params, result_dir_random)
    print("Finished. val acc:", random_result["best_val_acc"], "test acc", random_result["best_test_acc"])

    print(dataset, "nonnn", "transfer")
    result_dir_transfer = makedir([result_dir, "transfer"], remove_old_dir=True)
    tic = time.time()
    transfer = NonNNExperiment(data_dir, dataset, auto_prep_space, transfer_prep_path, baseline_prep_space, "tranfer", "rf")
    transfer_result = grid_search(transfer, params, result_dir_transfer)
    print("Finished in", time.time() - tic, "val acc:", transfer_result["best_val_acc"], "test acc", transfer_result["best_test_acc"])
