import numpy as np
import pandas as pd

import utils
from .experiment_utils import set_random_seed, load_data, build_data, grid_search, makedir
from model import LogisticRegression, TwoLayerNet
from pipeline.auto_prep_pipeline_order import AutoPrepPipeline as AutoPrepPipelineOrder
from pipeline.auto_prep_pipeline_order_feature import AutoPrepPipeline as AutoPrepPipelineOrderFeature
from pipeline.auto_prep_pipeline import AutoPrepPipeline
import torch
import torch.nn as nn
from trainer.auto_prep_batch import AutoPrepBatch
from trainer.auto_prep_sgd import AutoPrepSGD
from utils import SummaryWriter


class AutoPrepExperiment(object):
    """Run auto prep with one set of hyper parameters"""
    def __init__(self, data_dir, dataset, prep_space, model_name):
        self.data_dir = data_dir
        self.dataset = dataset
        self.prep_space = prep_space
        self.model_name = model_name

    def run(self, params):
        # load data
        set_random_seed(params)
        
        X, y = load_data(self.data_dir, self.dataset, keepna=False)
        X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=params["split_seed"], normalize=params["pre_norm"])

        # set random seed
        set_random_seed(params)

        ## transform pipeline
        # define and fit first step
        if params["order"]:
            prep_pipeline = AutoPrepPipelineOrder(self.prep_space, temperature=params["temperature"],
                                                    use_sample=params["sample"],
                                                    diff_method=params["diff_method"],
                                                    init_method=params["init_method"])
        elif params["order_feature"]:
            prep_pipeline = AutoPrepPipelineOrderFeature(self.prep_space, temperature=params["temperature"],
                                                    use_sample=params["sample"],
                                                    diff_method=params["diff_method"],
                                                    init_method=params["init_method"])
        else:
            prep_pipeline = AutoPrepPipeline(self.prep_space, temperature=params["temperature"],
                                             use_sample=params["sample"],
                                             diff_method=params["diff_method"],
                                             init_method=params["init_method"])

        prep_pipeline.init_parameters(X_train, X_val, X_test)
        print("Train size: ({}, {})".format(X_train.shape[0], prep_pipeline.out_features))

        # model
        input_dim = prep_pipeline.out_features
        output_dim = len(set(y.values.ravel()))

        # model = TwoLayerNet(input_dim, output_dim)
        set_random_seed(params)
        if self.model_name == "log":
            model = LogisticRegression(input_dim, output_dim)
        else:
            model = TwoLayerNet(input_dim, output_dim)

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

        prep_pipeline_optimizer = torch.optim.Adam(
            prep_pipeline.parameters(),
            lr=params["prep_lr"],
            betas=(0.5, 0.999),
            weight_decay=params["weight_decay"]
        )

        # scheduler
        # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=patience, factor=0.1, threshold=0.001)
        prep_pipeline_scheduler = None
        model_scheduler = None

        if params["logging"]:
            logger = SummaryWriter()
        else:
            logger = None

        auto_prep = AutoPrepSGD(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                    model_scheduler, prep_pipeline_scheduler, params, writer=logger)

        result, best_model = auto_prep.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        return result, best_model, logger

def run_auto_prep(data_dir, dataset, result_dir, prep_space, params, model_name):
    print("Dataset:", dataset, "Diff Method:", params["diff_method"], "First order:", params["first_order_only"])

    sample = "sample" if params["sample"] else "nosample"
    first_order = "first_order" if params["first_order_only"] else "second_order"
    method_name = "{}_{}_{}".format(sample, params["diff_method"], first_order)
    result_dir = makedir([result_dir, method_name], remove_old_dir=True)
  
    auto_prep_exp = AutoPrepExperiment(data_dir, dataset, prep_space, model_name)
    result = grid_search(auto_prep_exp, params, result_dir, save_model=params["save_model"])

    print("AutoPrep Finished. val acc:", result["best_val_acc"], "test acc", result["best_test_acc"])