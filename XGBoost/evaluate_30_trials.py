import argparse
import math
import time

import numpy as np
import optuna
import scipy
import zero
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

import lib
import wandb

from sklearn.model_selection import StratifiedKFold
from utils import set_random_seed

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--experiment_name', type=str, default='test',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--dataset', type=int, default=54,
                    help='The dataset ID to use. Default is 45068 (adult).')
parser.add_argument('--seed', type=int, default=0,
                    help='The random seed for reproducibility. Default is 42.')
parser.add_argument('--normalization', type=str, default='quantile', choices=['quantile', 'standard'],
                    help='The normalization to use for the numerical features. Default is "quantile".')
parser.add_argument('--cat_nan_policy', type=str, default='new', choices=['new', 'most_frequent'],
                    help='The policy to use for handling nan values in categorical features. Default is "new".')
parser.add_argument('--cat_policy', type=str, default='indices', choices=['indices', 'ohe'],
                    help='The policy to use for handling categorical features. Default is "indices".')
parser.add_argument('--outer_fold', type=int, default=0, help='The outer fold to use. Default is 0')
parser.add_argument('--n_trials', type=int, default=100,
                    help='The number of trials to use for HPO. Default is 100')
parser.add_argument('--tune', action='store_true', help='Whether to tune the hyperparameters using Optuna')

args = parser.parse_args()


def load_best_config(project_name, dataset_name, outer_fold, num_trials=30):
    api = wandb.Api()
    target_run_name = f"{dataset_name}_outerFold_{outer_fold}"
    runs = api.runs(project_name)

    target_run = None
    for run in runs:
        if run.name == target_run_name:
            target_run = run
            break

    if not target_run:
        raise ValueError(f"No run found with name: {target_run_name}")

    # First scan for the best average_test_rocauc
    best_rocauc = 0  # Looking for the highest rocauc
    best_step = None
    history = target_run.scan_history(keys=['average_test_rocauc'])
    for i, row in enumerate(history):
        if i >= num_trials:
            break
        if 'average_test_rocauc' in row and row['average_test_rocauc'] > best_rocauc:
            best_rocauc = row['average_test_rocauc']
            best_step = i

    if best_step is None:
        raise ValueError("Best rocauc not found within the first 30 trials")

    # Second scan for the HPs at the best step
    hp_keys = ['max_depth', 'min_child_weight', 'subsample', 'learning_rate', 'colsample_bylevel', 'colsample_bytree',
               'gamma', 'reg_lambda', 'reg_alpha']
    best_config = None
    history = target_run.scan_history(keys=hp_keys)
    for i, row in enumerate(history):
        if i == best_step:
            best_config = {key: row[key] for key in hp_keys if key in row}
            break

    if best_config:
        return best_config
    else:
        raise ValueError("HPs not found for the best rocauc step")


def run_single_outer_fold(outer_fold, D, outer_folds):
    outer_train_idx, outer_test_idx = outer_folds[outer_fold]

    best_params = load_best_config('t4tab/XGBoost_optuna', D.info['dataset_name'], args.outer_fold)

    hyperparameters = {
        'max_depth': best_params['max_depth'],
        'min_child_weight': best_params['min_child_weight'],
        'subsample': best_params['subsample'],
        'learning_rate': best_params['learning_rate'],
        'colsample_bylevel': best_params['colsample_bylevel'],
        'colsample_bytree': best_params['colsample_bytree'],
        'gamma': best_params['gamma'],
        'reg_lambda': best_params['reg_lambda'],
        'reg_alpha': best_params['reg_alpha']
    }
    X_outer_preprocessed = D.build_X(
        normalization='quantile',
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy='ohe',
        seed=args.seed,
        train_idx=outer_train_idx,
        test_idx=outer_test_idx,
    )
    set_random_seed(args.seed)
    Y, y_info = D.build_y(train_idx=outer_train_idx, test_idx=outer_test_idx)

    booster = "gbtree"
    early_stopping_rounds = 50
    n_estimators = 2000
    eval_metric = 'auc'
    model = XGBClassifier(booster=booster,
                          n_estimators=n_estimators,
                          tree_method='gpu_hist',
                          disable_default_eval_metric=True,
                          use_label_encoder=False)
    if args.tune:
        model.set_params(**hyperparameters)
    unique_classes, class_counts = np.unique(Y[outer_train_idx], axis=0, return_counts=True)
    nr_classes = len(unique_classes)

    model.fit(X_outer_preprocessed[outer_train_idx], Y[outer_train_idx],
              eval_set=[(X_outer_preprocessed[outer_test_idx], Y[outer_test_idx])],
              eval_metric=custom_auc_eval if D.is_multiclass else eval_metric,
              early_stopping_rounds=early_stopping_rounds,
              verbose=False)

    train_predictions_labels = model.predict(X_outer_preprocessed[outer_train_idx])
    test_predictions_labels = model.predict(X_outer_preprocessed[outer_test_idx])
    if D.is_multiclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_train_idx])
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_test_idx])
    else:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_train_idx])[:, 1]
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_test_idx])[:, 1]

    # calculate the balanced accuracy
    train_rocauc = roc_auc_score(Y[outer_train_idx], train_predictions_probabilities,
                                 multi_class='raise' if nr_classes == 2 else 'ovo')
    train_accuracy = accuracy_score(Y[outer_train_idx], train_predictions_labels)
    test_rocauc = roc_auc_score(Y[outer_test_idx], test_predictions_probabilities,
                                multi_class='raise' if nr_classes == 2 else 'ovo')
    test_accuracy = accuracy_score(Y[outer_test_idx], test_predictions_labels)
    print(f"Finished outer fold {outer_fold}")

    output_info = {
        'train_rocauc': train_rocauc,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        f'best_test_rocauc_outer_fold_{outer_fold}': test_rocauc,
    }
    wandb.log(output_info)
    wandb.finish()


def custom_auc_eval(y_pred, dtrain):
    y_true = dtrain.get_label()

    y_pred = scipy.special.softmax(y_pred, axis=1)
    y_pred_sums = np.sum(y_pred, axis=1)
    if not np.allclose(y_pred_sums, 1.0):
        print("Probabilities do not sum to 1.0 for some instances.")
        y_pred = y_pred / y_pred_sums[:, np.newaxis]
    auc = roc_auc_score(y_true, y_pred, multi_class='ovo')

    return 'auc', auc


if __name__ == "__main__":
    # %%
    set_random_seed(args.seed)
    D = lib.Dataset.from_openml(args.dataset)
    run_name = f"{D.info['dataset_name']}_outerFold_{args.outer_fold}"
    wandb.init(project=args.experiment_name,
               name=run_name,
               config=args)
    outer_kfold = StratifiedKFold(n_splits=10, shuffle=True)
    outer_folds = list(outer_kfold.split(D.X, D.y))
    run_single_outer_fold(args.outer_fold, D, outer_folds)