import argparse
import time

import numpy as np
import optuna
import pandas as pd
import scipy
import torch.optim
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

import metrics
from tab_model import TabNetClassifier, TabNetRegressor
from metrics import Metric

import lib
import wandb

from sklearn.model_selection import StratifiedKFold, KFold
from utils import set_random_seed
from lib.util import get_categories

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
parser.add_argument('--outer_fold', type=int, default=7, help='The outer fold to use. Default is 0')
parser.add_argument('--n_trials', type=int, default=100,
                    help='The number of trials to use for HPO. Default is 100')
parser.add_argument('--tune', action='store_true', help='Whether to tune the hyperparameters using Optuna')
args = parser.parse_args()


class TimeLimitExceededError(Exception):
    pass


def objective(trial, X_outer, Y_outer, start_time, time_limit, n_inner_folds=9):
    # Search space
    n_a = trial.suggest_categorical('n_a', [8, 16, 24, 32, 64, 128])
    learning_rate = trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.02, 0.025])
    gamma = trial.suggest_categorical('gamma', [1.0, 1.2, 1.5, 2.0])
    n_steps = trial.suggest_categorical('n_steps', [3, 4, 5, 6, 7, 8, 9, 10])
    lambda_sparse = trial.suggest_categorical('lambda_sparse', [0, 0.000001, 0.0001, 0.001, 0.01, 0.1])
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    virtual_batch_size = trial.suggest_categorical('virtual_batch_size', [256, 512, 1024, 2048, 4096])
    decay_rate = trial.suggest_categorical('decay_rate', [0.4, 0.8, 0.9, 0.95])
    decay_iterations = trial.suggest_categorical('decay_iterations', [500, 2000, 8000, 10000, 20000])
    momentum = trial.suggest_categorical('momentum', [0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
    epochs = trial.suggest_int('epochs', 10, 500)

    hyperparameters = {
        'n_a': n_a,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'n_steps': n_steps,
        'lambda_sparse': lambda_sparse,
        'batch_size': batch_size,
        'virtual_batch_size': virtual_batch_size,
        'decay_rate': decay_rate,
        'decay_iterations': decay_iterations,
        'momentum': momentum,
        'epochs': epochs,
    }
    wandb.log(hyperparameters)

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
            cat_policy='indices',
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
        N, C = X
        n_num_features = 0 if N is None else N[train_idx].shape[1]
        n_cat_features = 0 if C is None else C[train_idx].shape[1]
        n_features = n_num_features + n_cat_features
        if N is None:
            assert C is not None
            X = pd.DataFrame(C, columns=range(n_features))
        elif C is None:
            assert N is not None
            X = pd.DataFrame(N, columns=range(n_features))
        else:
            X = pd.concat(
                [
                    pd.DataFrame(N, columns=range(n_num_features)),
                    pd.DataFrame(C, columns=range(n_num_features, n_features)),
                ],
                axis=1
            )
        cat_features = list(range(n_num_features, n_features))
        categorical_counts = [len(np.unique(X.iloc[train_idx, i])) for i in cat_features]

        unique_classes, class_counts = np.unique(Y[train_idx], axis=0, return_counts=True)
        nr_classes = len(unique_classes)
        batch_size = min(batch_size, len(train_idx))
        virtual_batch_size = min(virtual_batch_size, batch_size)
        tabnet_params = {
            "cat_idxs": cat_features,
            "cat_dims": categorical_counts,
            "seed": args.seed,
            "device_name": "cuda",
        }
        if D.is_regression:
            model = TabNetRegressor(n_a=n_a, optimizer_params=dict(lr=learning_rate),
                                    scheduler_params=dict(decay_rate=decay_rate, decay_iterations=decay_iterations),
                                    gamma=gamma, n_steps=n_steps, lambda_sparse=lambda_sparse,
                                    optimizer_fn=torch.optim.AdamW, momentum=momentum, **tabnet_params)
            model.fit(X.values, Y.reshape(-1, 1),
                      eval_metric=['rmse'],
                      batch_size=batch_size,
                      virtual_batch_size=virtual_batch_size,
                      max_epochs=epochs
                      )
        else:
            model = TabNetClassifier(n_a=n_a, optimizer_params=dict(lr=learning_rate),
                                     scheduler_params=dict(decay_rate=decay_rate, decay_iterations=decay_iterations),
                                     gamma=gamma, n_steps=n_steps, lambda_sparse=lambda_sparse,
                                     optimizer_fn=torch.optim.AdamW, momentum=momentum, **tabnet_params)

            model.fit(X.values, Y,
                      eval_metric=['auc'],
                      batch_size=batch_size,
                      virtual_batch_size=virtual_batch_size,
                      weights=1,
                      max_epochs=epochs)
        train_predictions_labels = model.predict(X.iloc[train_idx].values)
        test_predictions_labels = model.predict(X.iloc[test_idx].values)

        if D.is_multiclass:
            train_predictions_probabilities = model.predict_proba(X.iloc[train_idx].values)
            test_predictions_probabilities = model.predict_proba(X.iloc[test_idx].values)
        elif D.is_binclass:
            train_predictions_probabilities = model.predict_proba(X.iloc[train_idx].values)[:, 1]
            test_predictions_probabilities = model.predict_proba(X.iloc[test_idx].values)[:, 1]

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
            train_rmse = mean_squared_error(Y[train_idx], train_predictions_labels) ** 0.5
            test_rmse = mean_squared_error(Y[test_idx], test_predictions_labels) ** 0.5
            rmse_across_inner_folds.append(test_rmse)
            print(f'Train rmse: {train_rmse}, test rmse: {test_rmse}')
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
            'best_n_a': best_params['n_a'],
            'best_learning_rate': best_params['learning_rate'],
            'best_gamma': best_params['gamma'],
            'best_n_steps': best_params['n_steps'],
            'best_lambda_sparse': best_params['lambda_sparse'],
            'best_batch_size': best_params['batch_size'],
            'best_virtual_batch_size': best_params['virtual_batch_size'],
            'best_decay_rate': best_params['decay_rate'],
            'best_decay_iterations': best_params['decay_iterations'],
            'best_momentum': best_params['momentum'],
            'best_epochs': best_params['epochs']
        }
        wandb.log(best_configuration)
    else:
        best_configuration = {}

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
    categorical_counts = [len(np.unique(X_outer_preprocessed.iloc[outer_train_idx, i])) for i in cat_features]
    unique_classes, class_counts = np.unique(Y[outer_train_idx], axis=0, return_counts=True)
    nr_classes = len(unique_classes)
    tabnet_params = {
        "cat_idxs": cat_features,
        "cat_dims": categorical_counts,
        "seed": args.seed,
        "device_name": "cuda",
    }
    if best_configuration:
        if D.is_regression:
            model = TabNetRegressor(n_a=best_configuration['best_n_a'],
                                    optimizer_params=dict(lr=best_configuration['best_learning_rate']),
                                    scheduler_params=dict(
                                        decay_rate=best_configuration['best_decay_rate'],
                                        decay_iterations=best_configuration['best_decay_iterations']),
                                    gamma=best_configuration['best_gamma'],
                                    n_steps=best_configuration['best_n_steps'],
                                    lambda_sparse=best_configuration['best_lambda_sparse'],
                                    optimizer_fn=torch.optim.AdamW,
                                    momentum=best_configuration['best_momentum'],
                                    **tabnet_params)
            model.fit(X_outer_preprocessed.iloc[outer_train_idx].values, Y[outer_train_idx].reshape(-1, 1),
                      eval_metric=['rmse'],
                      batch_size=min(best_configuration['best_batch_size'], len(outer_train_idx)),
                      virtual_batch_size=min(best_configuration['best_virtual_batch_size'], len(outer_train_idx)),
                      max_epochs=best_configuration['best_epochs'],
                      )
        else:
            model = TabNetClassifier(n_a=best_configuration['best_n_a'],
                                     optimizer_params=dict(lr=best_configuration['best_learning_rate']),
                                     scheduler_params=dict(
                                         decay_rate=best_configuration['best_decay_rate'],
                                         decay_iterations=best_configuration['best_decay_iterations']),
                                     gamma=best_configuration['best_gamma'],
                                     n_steps=best_configuration['best_n_steps'],
                                     lambda_sparse=best_configuration['best_lambda_sparse'],
                                     optimizer_fn=torch.optim.AdamW,
                                     momentum=best_configuration['best_momentum'],
                                     **tabnet_params)

            model.fit(X_outer_preprocessed.iloc[outer_train_idx].values, Y[outer_train_idx],
                      eval_metric=['auc'],
                      batch_size=min(best_configuration['best_batch_size'], len(outer_train_idx)),
                      virtual_batch_size=min(best_configuration['best_virtual_batch_size'], len(outer_train_idx)),
                      weights=1,
                      max_epochs=best_configuration['best_epochs'])
    else:
        if D.is_regression:
            model = TabNetRegressor(**tabnet_params)
            model.fit(X_outer_preprocessed.iloc[outer_train_idx].values, Y[outer_train_idx].reshape(-1, 1),
                      eval_metric=['rmse'],
                      batch_size=min(1024, len(outer_train_idx)),
                      virtual_batch_size=min(128, len(outer_train_idx)))
        else:
            model = TabNetClassifier(**tabnet_params)
            model.fit(X_outer_preprocessed.iloc[outer_train_idx].values, Y[outer_train_idx],
                      eval_metric=['auc'],
                      batch_size=min(1024, len(outer_train_idx)),
                      virtual_batch_size=min(128, len(outer_train_idx)),
                      weights=1,
                      max_epochs=100)

    train_predictions_labels = model.predict(X_outer_preprocessed.iloc[outer_train_idx].values)
    test_predictions_labels = model.predict(X_outer_preprocessed.iloc[outer_test_idx].values)
    if D.is_multiclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_train_idx].values)
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_test_idx].values)
    elif D.is_binclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_train_idx].values)[:, 1]
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_test_idx].values)[:, 1]

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
