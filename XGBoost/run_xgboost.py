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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

import lib
import wandb

from sklearn.model_selection import StratifiedKFold, KFold
from utils import set_random_seed

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--experiment_name', type=str, default='test',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--dataset', type=int, default=11,
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


class TimeLimitExceededError(Exception):
    pass


def objective(trial, X_outer, Y_outer, start_time, time_limit, n_inner_folds=9):
    # Search space
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_child_weight = trial.suggest_float('min_child_weight', 1e-8, 1e5, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1.0, log=True)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 1e-8, 1e2, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 1e2, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 1e2, log=True)
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)

    hyperparameters = {
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'learning_rate': learning_rate,
        'colsample_bylevel': colsample_bylevel,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'n_estimators': n_estimators
    }
    wandb.log(hyperparameters)

    booster = "gbtree"
    eval_metric = 'rmse' if D.is_regression else 'auc'
    average_test_rocaucs = []
    rmse_across_inner_folds = []

    if D.is_regression:
        kfold = KFold(n_splits=n_inner_folds, shuffle=True, random_state=args.seed)
        split = kfold.split(X_outer)
    else:
        kfold = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=args.seed)
        split = kfold.split(X_outer, Y_outer)

    for fold, (train_idx, test_idx) in enumerate(split):
        if time.time() - start_time > time_limit:
            raise TimeLimitExceededError("Global time limit exceeded. Terminating HPO.")
        print("Running inner fold", fold)
        X = D.build_X(
            normalization='quantile',
            num_nan_policy='mean',
            cat_nan_policy='new',
            cat_policy='ohe',
            seed=args.seed,
            X=X_outer,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        set_random_seed(args.seed)
        if D.is_regression:
            y_policy = 'mean_std'
        else:
            y_policy = None
        Y, y_info = D.build_y(train_idx=train_idx, test_idx=test_idx, y=Y_outer, policy=y_policy)
        if isinstance(Y, dict):
            Y = {k: v.astype(float) for k, v in Y.items()}
            max_index = len(Y)
            Y_np_array = np.empty(max_index, dtype=np.float32)
            for i, idx in enumerate(Y):
                Y_np_array[i] = Y[idx].item()
            Y = Y_np_array
        if D.is_multiclass or D.is_binclass:
            model = XGBClassifier(booster=booster,
                                  n_estimators=n_estimators,
                                  tree_method='gpu_hist',
                                  disable_default_eval_metric=True,
                                  use_label_encoder=False)
        else:
            model = XGBRegressor(booster=booster,
                                 n_estimators=n_estimators,
                                 tree_method='gpu_hist',
                                 disable_default_eval_metric=True,
                                 use_label_encoder=False)
        model.set_params(**hyperparameters)
        unique_classes, class_counts = np.unique(Y[train_idx], axis=0, return_counts=True)
        nr_classes = len(unique_classes)

        model.fit(X[train_idx], Y[train_idx],
                  eval_metric=custom_auc_eval if D.is_multiclass else eval_metric,
                  verbose=False)
        train_predictions_labels = model.predict(X[train_idx])
        test_predictions_labels = model.predict(X[test_idx])

        if D.is_multiclass:
            train_predictions_probabilities = model.predict_proba(X[train_idx])
            test_predictions_probabilities = model.predict_proba(X[test_idx])
        elif D.is_binclass:
            train_predictions_probabilities = model.predict_proba(X[train_idx])[:, 1]
            test_predictions_probabilities = model.predict_proba(X[test_idx])[:, 1]

        if D.is_multiclass or D.is_binclass:
            train_rocauc = roc_auc_score(Y[train_idx], train_predictions_probabilities,
                                         multi_class='raise' if nr_classes == 2 else 'ovo')
            train_accuracy = accuracy_score(Y[train_idx], train_predictions_labels)
            test_rocauc = roc_auc_score(Y[test_idx], test_predictions_probabilities,
                                        multi_class='raise' if nr_classes == 2 else 'ovo')
            average_test_rocaucs.append(test_rocauc)
            test_accuracy = accuracy_score(Y[test_idx], test_predictions_labels)
            print(f'Train rocauc: {train_rocauc}, test rocauc: {test_rocauc}')
            print(f'Train accuracy: {train_accuracy}, test accuracy: {test_accuracy}')
        else:
            train_mse = mean_squared_error(Y[train_idx], train_predictions_labels) ** 0.5
            test_mse = mean_squared_error(Y[test_idx], test_predictions_labels) ** 0.5
            rmse_across_inner_folds.append(test_mse)
            print(f'Train mse: {train_mse}, test mse: {test_mse}')
    print('Finished inner folds')
    if D.is_regression:
        avg_mse_loss = sum(rmse_across_inner_folds) / len(rmse_across_inner_folds)
        print("Average mse loss across inner folds:", avg_mse_loss)
        optuna_metric = avg_mse_loss
        wandb.log({"avg_mse_loss": avg_mse_loss})
    else:
        average_test_rocauc_across_folds = np.mean(average_test_rocaucs)
        print("Average test rocauc across inner folds:", average_test_rocauc_across_folds)
        optuna_metric = 1 - average_test_rocauc_across_folds
        wandb.log({'average_test_rocauc': average_test_rocauc_across_folds})
    return optuna_metric


def run_single_outer_fold(outer_fold, D, outer_folds):
    outer_train_idx, outer_test_idx = outer_folds[outer_fold]
    time_limit = 23 * 60 * 60
    start_time = time.time()
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    if args.tune:
        try:
            study.optimize(
                lambda trial: objective(trial, D.X.iloc[outer_train_idx], D.y[outer_train_idx],
                                        start_time, time_limit),
                n_trials=args.n_trials, timeout=23 * 60 * 60)
        except TimeLimitExceededError:
            print("Optimization stopped due to global time limit.")
        best_params = study.best_params
        best_configuration = {
            'best_max_depth': best_params['max_depth'],
            'best_min_child_weight': best_params['min_child_weight'],
            'best_subsample': best_params['subsample'],
            'best_learning_rate': best_params['learning_rate'],
            'best_colsample_bylevel': best_params['colsample_bylevel'],
            'best_colsample_bytree': best_params['colsample_bytree'],
            'best_gamma': best_params['gamma'],
            'best_reg_lambda': best_params['reg_lambda'],
            'best_reg_alpha': best_params['reg_alpha'],
            'best_n_estimators': best_params['n_estimators']
        }
        hyperparameters = {
            'max_depth': best_params['max_depth'],
            'min_child_weight': best_params['min_child_weight'],
            'subsample': best_params['subsample'],
            'learning_rate': best_params['learning_rate'],
            'colsample_bylevel': best_params['colsample_bylevel'],
            'colsample_bytree': best_params['colsample_bytree'],
            'gamma': best_params['gamma'],
            'reg_lambda': best_params['reg_lambda'],
            'reg_alpha': best_params['reg_alpha'],
            'n_estimators': best_params['n_estimators']
        }
        wandb.log(best_configuration)
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
    if D.is_regression:
        y_policy = 'mean_std'
    else:
        y_policy = None
    Y, y_info = D.build_y(train_idx=outer_train_idx, test_idx=outer_test_idx, policy=y_policy)
    if isinstance(Y, dict):
        Y = {k: v.astype(float) for k, v in Y.items()}
        max_index = len(Y)
        Y_np_array = np.empty(max_index, dtype=np.float32)
        for i, idx in enumerate(Y):
            Y_np_array[i] = Y[idx].item()
        Y = Y_np_array
    booster = "gbtree"
    eval_metric = 'rmse' if D.is_regression else 'auc'
    if D.is_multiclass or D.is_binclass:
        model = XGBClassifier(booster=booster,
                              tree_method='gpu_hist',
                              disable_default_eval_metric=True,
                              use_label_encoder=False)
    else:
        model = XGBRegressor(booster=booster,
                             tree_method='gpu_hist',
                             disable_default_eval_metric=True,
                             use_label_encoder=False)
    if args.tune:
        model.set_params(**hyperparameters)
    unique_classes, class_counts = np.unique(Y[outer_train_idx], axis=0, return_counts=True)
    nr_classes = len(unique_classes)

    model.fit(X_outer_preprocessed[outer_train_idx], Y[outer_train_idx],
              eval_metric=custom_auc_eval if D.is_multiclass else eval_metric,
              verbose=False)

    train_predictions_labels = model.predict(X_outer_preprocessed[outer_train_idx])
    test_predictions_labels = model.predict(X_outer_preprocessed[outer_test_idx])
    if D.is_multiclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_train_idx])
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_test_idx])
    elif D.is_binclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_train_idx])[:, 1]
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed[outer_test_idx])[:, 1]

    if D.is_multiclass or D.is_binclass:
        train_rocauc = roc_auc_score(Y[outer_train_idx], train_predictions_probabilities,
                                     multi_class='raise' if nr_classes == 2 else 'ovo')
        train_accuracy = accuracy_score(Y[outer_train_idx], train_predictions_labels)
        test_rocauc = roc_auc_score(Y[outer_test_idx], test_predictions_probabilities,
                                    multi_class='raise' if nr_classes == 2 else 'ovo')
        test_accuracy = accuracy_score(Y[outer_test_idx], test_predictions_labels)
    else:
        train_rmse = mean_squared_error(Y[outer_train_idx], train_predictions_labels) ** 0.5
        test_rmse = mean_squared_error(Y[outer_test_idx], test_predictions_labels) ** 0.5
        print(f'Train mse: {train_rmse}, test mse: {test_rmse}')
    print(f"Finished outer fold {outer_fold}")

    if D.is_multiclass or D.is_binclass:
        output_info = {
            'train_rocauc': train_rocauc,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            f'best_test_rocauc_outer_fold_{outer_fold}': test_rocauc,
        }
    else:
        output_info = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            f'best_test_rmse_outer_fold_{outer_fold}': test_rmse,
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
    if D.is_regression:
        outer_kfold = KFold(n_splits=10, shuffle=True, random_state=args.seed)
        outer_folds = list(outer_kfold.split(D.X))
    else:
        outer_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        outer_folds = list(outer_kfold.split(D.X, D.y))
    run_single_outer_fold(args.outer_fold, D, outer_folds)
