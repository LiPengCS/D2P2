import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
import shutil
from itertools import product
from copy import deepcopy

import utils
from sklearn.preprocessing import MinMaxScaler

def makedir(dir_list, file=None, remove_old_dir=False):
    save_dir = os.path.join(*dir_list)

    if remove_old_dir and os.path.exists(save_dir) and file is None:
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

def load_df(file_path, dataset_info):
    if "missing_value" in dataset_info.keys():
        na_values = dataset_info["missing_value"]
    else:
        na_values = None

    df = pd.read_csv(file_path, na_values=na_values)

    if "drop_variables" in dataset_info.keys():
        df = df.drop(columns=dataset_info["drop_variables"])

    if 'categorical_variables' in dataset_info.keys():
        categories = dataset_info['categorical_variables']
        if categories == "all":
            for cat in df.columns:
                df[cat] = df[cat].astype(str).replace('nan', np.nan)
        else:
            for cat in categories:
                df[cat] = df[cat].astype(str).replace('nan', np.nan)
    return df

def split(X, y, val_ratio=0.2, test_ratio=0.2, random_state=1):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test 

def build_data(X, y, random_state=1, normalize=False):
    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y.values.ravel())
    y_enc = torch.tensor(y_enc).long()
    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y_enc, random_state=random_state)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    if normalize:
        num_columns = X_train.select_dtypes(include='number').columns
        cat_columns = X_train.select_dtypes(exclude='number').columns

        X_train_num = X_train[num_columns]
        X_val_num = X_val[num_columns]
        X_test_num = X_test[num_columns]
        X_train_cat = X_train[cat_columns]
        X_val_cat = X_val[cat_columns]
        X_test_cat = X_test[cat_columns]

        scaler = MinMaxScaler()
        X_train_num_norm = pd.DataFrame(scaler.fit_transform(X_train_num.values), columns=num_columns)
        X_val_num_norm = pd.DataFrame(scaler.transform(X_val_num.values), columns=num_columns)
        X_test_num_norm = pd.DataFrame(scaler.transform(X_test_num.values), columns=num_columns)

        X_train = pd.concat([X_train_num_norm, X_train_cat], axis=1)[X_train.columns]
        X_val = pd.concat([X_val_num_norm, X_val_cat], axis=1)[X_train.columns]
        X_test = pd.concat([X_test_num_norm, X_test_cat], axis=1)[X_train.columns]

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_info(info_dir):
    info_path = os.path.join(info_dir, "info.json")
    with open(info_path) as info_data:
        info = json.load(info_data)
    return info

def load_data(data_dir, dataset, keepna=True):
    # load info dict
    dataset_dir = os.path.join(data_dir, dataset)
    info = load_info(dataset_dir)

    file_path = os.path.join(dataset_dir, "data.csv")
    data = load_df(file_path, info)

    if not keepna:
        data = data.dropna(how='all').reset_index(drop=True)

    label_column = info["label"]
    feature_column = [c for c in data.columns if c != label_column]
    X = data[feature_column]
    y = data[[label_column]]
    return X, y

def set_random_seed(params):
    random_state = params["seed"]
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if "cuda" in params["device"]:
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

def save_result(result, model_dict, logger, params, save_dir, save_model=False):
    # save logger
    logger.save(utils.makedir([save_dir, "logging"]))

    # save params and results
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

    # save model
    if save_model and model_dict is not None:
        for name, model in model_dict.items():
            torch.save(model, os.path.join(save_dir, "{}.pth".format(name)))

def copy_result(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def grid_search(experiment_obj, param_grid, result_dir, save_all=False, save_model=False):
    # best acc with early stopping
    best_result = None
    best_val_loss = float("inf")
    temp_dir = None

    for i, params in enumerate(get_param_candidates(param_grid)):
        # print("Model lr {}".format(model_lr))
        if "no_crash" in params and params["no_crash"]:
            try:
                result, model, logger = experiment_obj.run(params)
            except:
                print("Error!!!!!!")
                continue
        else:
            result, model, logger = experiment_obj.run(params)


        temp_dir = makedir([result_dir, "temp"], remove_old_dir=True)
        save_result(result, model, logger, params, temp_dir, save_model)

        if save_all:
            save_dir = makedir([result_dir, "all_results", str(i)], remove_old_dir=True)
            copy_result(temp_dir, save_dir)

        # if result["best_val_acc"] > best_val_acc:
        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_result = result

            save_dir = makedir([result_dir, "best_best"], remove_old_dir=True)
            copy_result(temp_dir, save_dir)

    # remove temp dir
    if temp_dir is not None and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return best_result

def get_param_candidates(param_grid):
    fixed_params = {}
    tuned_params = {}
    candidate_params = []

    for name, parameter in param_grid.items():
        if type(parameter) == list:
            tuned_params[name] = parameter
        else:
            fixed_params[name] = parameter

    for tuned_params_cand in product(*tuned_params.values()):
        param_cand = deepcopy(fixed_params)
        for n, p in zip(tuned_params.keys(), tuned_params_cand):
            param_cand[n] = p
        candidate_params.append(param_cand)

    return candidate_params


def print_params(model):
    for name, w in model.named_parameters():
        print(name, w.shape)