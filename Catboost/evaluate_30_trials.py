import argparse
import math
import time

import numpy as np
import optuna
import pandas as pd
import zero
import torch.nn as nn
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier
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
parser.add_argument('--dataset', type=int, default=23,
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
    hp_keys = ['max_depth', 'learning_rate', 'bagging_temperature', 'l2_leaf_reg', 'leaf_estimation_iterations']
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

    best_configuration = load_best_config('t4tab/CatboostFT_optuna_CPU', D.info['dataset_name'], args.outer_fold)

    X_outer_preprocessed = D.build_X(
        normalization='quantile',
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy='indices',
        seed=args.seed,
        train_idx=outer_train_idx,
        test_idx=outer_test_idx,
    )
    set_random_seed(args.seed)
    Y, y_info = D.build_y(train_idx=outer_train_idx, test_idx=outer_test_idx)

    N, C = X_outer_preprocessed
    n_num_features = 0 if N is None else N[outer_train_idx].shape[1]
    n_cat_features = 0 if C is None else C[outer_train_idx].shape[1]
    n_features = n_num_features + n_cat_features
    if N is None:
        assert C is not None
        X_outer_preprocessed = pd.DataFrame(C, columns=range(n_features))
    elif C is None:
        assert N is not None
        X_outer_preprocessed = pd.DataFrame(N, columns=range(n_features))
    else:
        X_outer_preprocessed = pd.concat(
            [
                pd.DataFrame(N, columns=range(n_num_features)),
                pd.DataFrame(C, columns=range(n_num_features, n_features)),
            ],
            axis=1
        )
    cat_features = list(range(n_num_features, n_features))
    unique_classes, class_counts = np.unique(Y[outer_train_idx], axis=0, return_counts=True)
    nr_classes = len(unique_classes)
    model = CatBoostClassifier(
        task_type='CPU',
        loss_function='MultiClass' if nr_classes > 2 else 'Logloss',
        eval_metric='AUC',
        random_seed=args.seed,
        early_stopping_rounds=50,
        od_pval=0.001,
        iterations=2000,
        max_depth=best_configuration['max_depth'],
        learning_rate=best_configuration['learning_rate'],
        bagging_temperature=best_configuration['bagging_temperature'],
        l2_leaf_reg=best_configuration['l2_leaf_reg'],
        leaf_estimation_iterations=best_configuration['leaf_estimation_iterations'],
    )

    model.fit(X_outer_preprocessed.iloc[outer_train_idx], Y[outer_train_idx],
              eval_set=(X_outer_preprocessed.iloc[outer_test_idx], Y[outer_test_idx]),
              cat_features=cat_features,
              verbose=False)

    train_predictions_labels = model.predict(X_outer_preprocessed.iloc[outer_train_idx])
    test_predictions_labels = model.predict(X_outer_preprocessed.iloc[outer_test_idx])
    if D.is_multiclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_train_idx])
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_test_idx])
    else:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_train_idx])[:, 1]
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_test_idx])[:, 1]

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
