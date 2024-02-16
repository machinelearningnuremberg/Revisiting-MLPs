import argparse
import os
import time
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

import lib
import wandb

from sklearn.model_selection import StratifiedKFold, KFold
from utils import set_random_seed

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--experiment_name', type=str, default='test',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--dataset', type=int, default=3,
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
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 1, log=True)
    bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10, log=True)
    leaf_estimation_iterations = trial.suggest_int('leaf_estimation_iterations', 1, 10)
    iterations = trial.suggest_int('iterations', 100, 2000)
    hyperparameters = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'bagging_temperature': bagging_temperature,
        'l2_leaf_reg': l2_leaf_reg,
        'leaf_estimation_iterations': leaf_estimation_iterations,
        'iterations': iterations,
    }
    wandb.log(hyperparameters)

    average_test_rocaucs = []
    mse_across_inner_folds = []

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
        unique_classes, class_counts = np.unique(Y[train_idx], axis=0, return_counts=True)
        nr_classes = len(unique_classes)
        if D.is_regression:
            model = CatBoostRegressor(
                task_type='CPU',
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=args.seed,
                iterations=iterations,
                max_depth=max_depth,
                learning_rate=learning_rate,
                bagging_temperature=bagging_temperature,
                l2_leaf_reg=l2_leaf_reg,
                leaf_estimation_iterations=leaf_estimation_iterations,
            )
        else:
            model = CatBoostClassifier(
                task_type='CPU',
                loss_function='MultiClass' if nr_classes > 2 else 'Logloss',
                eval_metric='AUC',
                random_seed=args.seed,
                iterations=iterations,
                max_depth=max_depth,
                learning_rate=learning_rate,
                bagging_temperature=bagging_temperature,
                l2_leaf_reg=l2_leaf_reg,
                leaf_estimation_iterations=leaf_estimation_iterations,
            )

        model.fit(X.iloc[train_idx], Y[train_idx],
                  cat_features=cat_features,
                  verbose=False)
        train_predictions_labels = model.predict(X.iloc[train_idx])
        test_predictions_labels = model.predict(X.iloc[test_idx])

        if D.is_multiclass:
            train_predictions_probabilities = model.predict_proba(X.iloc[train_idx])
            test_predictions_probabilities = model.predict_proba(X.iloc[test_idx])
        elif D.is_binclass:
            train_predictions_probabilities = model.predict_proba(X.iloc[train_idx])[:, 1]
            test_predictions_probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]

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
            mse_across_inner_folds.append(test_rmse)
            print(f'Train rmse: {train_rmse}, test rmse: {test_rmse}')


    print('Finished inner folds')
    if D.is_regression:
        avg_mse_loss = sum(mse_across_inner_folds) / len(mse_across_inner_folds)
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
            'best_learning_rate': best_params['learning_rate'],
            'best_bagging_temperature': best_params['bagging_temperature'],
            'best_l2_leaf_reg': best_params['l2_leaf_reg'],
            'best_leaf_estimation_iterations': best_params['leaf_estimation_iterations'],
            'best_iterations': best_params['iterations'],
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
    unique_classes, class_counts = np.unique(Y[outer_train_idx], axis=0, return_counts=True)
    nr_classes = len(unique_classes)
    if D.is_regression:
        model = CatBoostRegressor(
            task_type='CPU',
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=args.seed,
            iterations=best_configuration.get('best_iterations', None),
            max_depth=best_configuration.get('best_max_depth', None),
            learning_rate=best_configuration.get('best_learning_rate', None),
            bagging_temperature=best_configuration.get('best_bagging_temperature', None),
            l2_leaf_reg=best_configuration.get('best_l2_leaf_reg', None),
            leaf_estimation_iterations=best_configuration.get('best_leaf_estimation_iterations', None),
        )
    else:
        model = CatBoostClassifier(
            task_type='CPU',
            loss_function='MultiClass' if nr_classes > 2 else 'Logloss',
            eval_metric='AUC',
            random_seed=args.seed,
            iterations=best_configuration.get('best_iterations', None),
            max_depth=best_configuration.get('best_max_depth', None),
            learning_rate=best_configuration.get('best_learning_rate', None),
            bagging_temperature=best_configuration.get('best_bagging_temperature', None),
            l2_leaf_reg=best_configuration.get('best_l2_leaf_reg', None),
            leaf_estimation_iterations=best_configuration.get('best_leaf_estimation_iterations', None),
        )

    model.fit(X_outer_preprocessed.iloc[outer_train_idx], Y[outer_train_idx],
              cat_features=cat_features,
              verbose=False)

    train_predictions_labels = model.predict(X_outer_preprocessed.iloc[outer_train_idx])
    test_predictions_labels = model.predict(X_outer_preprocessed.iloc[outer_test_idx])
    if D.is_multiclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_train_idx])
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_test_idx])
    elif D.is_binclass:
        train_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_train_idx])[:, 1]
        test_predictions_probabilities = model.predict_proba(X_outer_preprocessed.iloc[outer_test_idx])[:, 1]

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



if __name__ == "__main__":
    # %%
    set_random_seed(args.seed)
    start_time = time.time()
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
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    wandb.log({'total_time': end_time - start_time})
    wandb.finish()
