import argparse
import math
import os
import time
import pickle

import optuna
import zero
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.data import set_random_seed
from resnext import ResNext

import lib
import wandb

from sklearn.model_selection import StratifiedKFold, KFold
from resnet_ft import ResNet

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--batch_size', type=int, default=512,
                    help='The batch size for training. Default is 32.')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--eval_batch_size', type=int, default=8192,
                    help='The batch size for evaluation. Default is 8192.')
parser.add_argument('--dataset', type=int, default=3,
                    help='The dataset ID to use. Default is 45068 (adult).')
parser.add_argument('--seed', type=int, default=0,
                    help='The random seed for reproducibility. Default is 42.')
parser.add_argument('--layer_size', type=int, default=128,
                    help='The layer size to use. Default is 512.')
parser.add_argument('--n_layers', type=int, default=2,
                    help='The number of blocks to use. Default is 2.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model for. Default is 100.')
parser.add_argument('--patience', type=int, default=16,
                    help='Number of epochs to wait for improvement before early stopping. Default is 16.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='The learning rate to use. Default is 1e-4.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='The weight decay to use. Default is 1e-5.')
parser.add_argument('--residual_dropout', type=float, default=0.2,
                    help='The dropout to use for the residual connections. Default is 0.0.')
parser.add_argument('--hidden_dropout', type=float, default=0.2,
                    help='The dropout to use for the hidden layers. Default is 0.0.')
parser.add_argument('--d_embedding', type=int, default=64,
                    help='The embedding size to use for categorical features. Default is 64.')
parser.add_argument('--d_hidden_factor', type=float, default=1.0,
                    help='The hidden factor to use for the hidden layers. Default is 1.0.')
parser.add_argument('--normalization', type=str, default='quantile', choices=['quantile', 'standard'],
                    help='The normalization to use for the numerical features. Default is "quantile".')
parser.add_argument('--cat_nan_policy', type=str, default='new', choices=['new', 'most_frequent'],
                    help='The policy to use for handling nan values in categorical features. Default is "new".')
parser.add_argument('--cat_policy', type=str, default='indices', choices=['indices', 'ohe'],
                    help='The policy to use for handling categorical features. Default is "indices".')
parser.add_argument('--outer_fold', type=int, default=0, help='The outer fold to use. Default is 0')
parser.add_argument('--n_trials', type=int, default=2,
                    help='The number of trials to use for HPO. Default is 100')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'resnext'],
                    help='The model to use. Default is "resnet".')
parser.add_argument('--cardinality', type=int, default=4, choices=[2, 4, 8, 16, 32],
                    help='The cardinality to use for ResNext. Default is 4.')

parser.add_argument('--tune', action='store_true', help='Whether to tune the hyperparameters using Optuna')

args = parser.parse_args()


def objective(trial, X_outer, Y_outer, batch_size, eval_batch_size, start_time, time_limit, n_inner_folds=9):
    # Search space
    cardinality = None

    layer_size = trial.suggest_int('layer_size', 64, 1024)
    if args.model == 'resnext':
        cardinality = trial.suggest_categorical('cardinality', [2, 4, 8, 16, 32])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    residual_dropout = trial.suggest_float('residual_dropout', 0.0, 0.5)
    hidden_dropout = trial.suggest_float('hidden_dropout', 0.0, 0.5)
    n_layers = trial.suggest_int('n_layers', 1, 8)
    d_embedding = trial.suggest_int('d_embedding', 64, 512)
    d_hidden_factor = trial.suggest_float('d_hidden_factor', 1.0, 4.0)
    epochs = trial.suggest_int('epochs', 10, 500)

    hyperparameters = {
        "layer_size": layer_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "residual_dropout": residual_dropout,
        "hidden_dropout": hidden_dropout,
        "n_layers": n_layers,
        "d_embedding": d_embedding,
        "d_hidden_factor": d_hidden_factor,
        "cardinality": cardinality if args.model == 'resnext' else None,
        "epochs": epochs

    }
    wandb.log(hyperparameters)

    rocauc_across_inner_folds = []
    mse_across_inner_folds = []

    best_mse_loss = 1e10

    if D.is_regression:
        kfold = KFold(n_splits=n_inner_folds, shuffle=True, random_state=args.seed)
        split = kfold.split(X_outer)
    else:
        kfold = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=args.seed)
        split = kfold.split(X_outer, Y_outer)

    for fold, (train_idx, test_idx) in enumerate(split):
        print("Running inner fold", fold)
        X = D.build_X(
            normalization=args.normalization,
            num_nan_policy='mean',
            cat_nan_policy='new',
            cat_policy=args.cat_policy,
            seed=args.seed,
            X=X_outer,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        if not isinstance(X, tuple):
            X = (X, None)
        set_random_seed(args.seed)
        if D.is_regression:
            y_policy = 'mean_std'
        else:
            y_policy = None
        Y, y_info = D.build_y(train_idx=train_idx, test_idx=test_idx, y=Y_outer, policy=y_policy)
        X = tuple(None if x is None else lib.to_tensors(x) for x in X)
        Y = lib.to_tensors(Y)
        device = lib.get_device()
        if device.type != 'cpu':
            X = tuple(
                None if x is None else x.to(device) for x in X
            )
            if isinstance(Y, dict):
                Y_device = {k: v.to(device) for k, v in Y.items()}
            else:
                Y_device = Y.to(device)
        else:
            Y_device = Y
        X_num, X_cat = X
        if X_cat is not None:
            X_cat = X_cat.long()
        del X
        if not D.is_multiclass:
            if isinstance(Y_device, dict):
                Y_device = {k: v.float() for k, v in Y_device.items()}
                max_index = len(Y_device)
                Y_tensor = torch.empty(max_index, device='cuda:0')
                for i, idx in enumerate(Y_device):
                    Y_tensor[i] = Y_device[idx]
                Y_device = Y_tensor
            else:
                Y_device = Y_device.float()
        else:
            Y_device = Y_device.long()

        train_size = D.size(train_idx)
        epoch_size = math.ceil(train_size / batch_size)
        chunk_size = None

        loss_fn = (
            F.binary_cross_entropy_with_logits
            if D.is_binclass
            else F.cross_entropy
            if D.is_multiclass
            else F.mse_loss
        )
        if args.model == 'resnet':

            model = ResNet(
                d_numerical=0 if X_num is None else X_num[train_idx].shape[1],
                categories=lib.get_categories(X_cat, train_idx, test_idx),
                d_out=D.info['n_classes'] if D.is_multiclass else 1,
                d=layer_size,
                n_layers=n_layers,
                hidden_dropout=hidden_dropout,
                residual_dropout=residual_dropout,
                d_embedding=d_embedding,
                d_hidden_factor=d_hidden_factor,
                activation='relu',
                normalization='batchnorm',
            ).to(device)
        else:
            model = ResNext(
                d_numerical=0 if X_num is None else X_num[train_idx].shape[1],
                categories=lib.get_categories(X_cat, train_idx, test_idx),
                d_out=D.info['n_classes'] if D.is_multiclass else 1,
                d=layer_size,
                n_layers=n_layers,
                hidden_dropout=hidden_dropout,
                residual_dropout=residual_dropout,
                d_embedding=d_embedding,
                d_hidden_factor=d_hidden_factor,
                activation='relu',
                normalization='batchnorm',
                cardinality=cardinality,
            ).to(device)
        if torch.cuda.device_count() > 1:  # type: ignore[code]
            print('Using nn.DataParallel')
            model = nn.DataParallel(model)
        num_parameters = lib.get_n_parameters(model)
        optimizer = lib.make_optimizer(
            "adamw",
            (
                [
                    {'params': model.parameters()},
                ]
            ),
            lr,
            weight_decay,
        )

        stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
        progress = zero.ProgressTracker(args.patience)
        training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
        timer = zero.Timer()

        def print_epoch_info():
            print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())}')
            print(
                ' | '.join(
                    f'{k} = {v}'
                    for k, v in {
                        'lr': lib.get_lr(optimizer),
                        'batch_size': batch_size,
                        'chunk_size': chunk_size,
                        'epoch_size': epoch_size,
                        'n_parameters': num_parameters,
                    }.items()
                )
            )

        def apply_model(indices, idx):
            return model(
                None if X_num is None else X_num[indices][idx],
                None if X_cat is None else X_cat[indices][idx],
            )

        @torch.no_grad()
        def evaluate(indices, eval_batch_size):
            model.eval()
            metrics = {}
            predictions = {}
            while eval_batch_size:
                try:
                    predictions["test"] = (
                        torch.cat(
                            [
                                apply_model(indices, idx)
                                for idx in lib.IndexLoader(
                                D.size(indices), eval_batch_size, False, device
                            )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
            metrics['test'] = lib.calculate_metrics(
                D.info['task_type'],
                Y_device[indices].cpu().numpy(),  # type: ignore[code]
                predictions['test'],  # type: ignore[code]
                'logits',
                y_info,
            )
            for part_metrics in metrics.items():
                print(f'[test]', lib.make_summary(part_metrics))
            return metrics, predictions

        # %%
        timer.run()
        for epoch in stream.epochs(epochs):
            if time.time() - start_time > time_limit:
                raise optuna.exceptions.OptunaError("Time limit exceeded, terminating the HPO process.")
            print_epoch_info()

            model.train()
            epoch_losses = []
            for batch_idx in epoch:
                loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                    optimizer,
                    loss_fn,
                    lambda x: (apply_model(train_idx, x), Y_device[train_idx][x]),
                    batch_idx,
                    chunk_size or batch_size,
                )
                epoch_losses.append(loss.detach())
                if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                    chunk_size = new_chunk_size
                    print('New chunk size:', chunk_size)
            epoch_losses = torch.stack(epoch_losses).tolist()
            training_log[lib.TRAIN].extend(epoch_losses)
            print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate(test_idx, eval_batch_size)
        if D.is_regression:
            mse_across_inner_folds.append(best_mse_loss)
        else:
            rocauc_across_inner_folds.append(metrics['test']['score'])

    print("Finished inner folds")
    if D.is_regression:
        avg_mse_loss = sum(mse_across_inner_folds) / len(mse_across_inner_folds)
        print("Average mse loss across inner folds:", avg_mse_loss)
        optuna_metric = avg_mse_loss
        wandb.log({"avg_mse_loss": avg_mse_loss})
    else:
        avg_test_rocauc = sum(rocauc_across_inner_folds) / len(rocauc_across_inner_folds)
        print("Average rocauc across inner folds:", avg_test_rocauc)
        optuna_metric = 1 - avg_test_rocauc
        wandb.log({"avg_test_rocauc": avg_test_rocauc})

    return optuna_metric


def run_single_outer_fold(outer_fold, D, outer_folds):
    outer_train_idx, outer_test_idx = outer_folds[outer_fold]
    # batch size settings for datasets in (Grinsztajn et al., 2022)
    if D.n_features <= 32:
        batch_size = 512
        eval_batch_size = 8192
    elif D.n_features <= 100:
        batch_size = 128
        eval_batch_size = 512
    elif D.n_features <= 1000:
        batch_size = 32
        eval_batch_size = 64
    else:
        batch_size = 16
        eval_batch_size = 16
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    time_limit = 23 * 60 * 60
    start_time = time.time()
    if args.tune:
        try:
            study.optimize(
                lambda trial: objective(trial, D.X.iloc[outer_train_idx], D.y[outer_train_idx], batch_size,
                                        eval_batch_size, start_time, time_limit), n_trials=args.n_trials)
        except optuna.exceptions.OptunaError as e:
            print(f"Optimization stopped: {e}")
        best_params = study.best_params
        best_configuration = {
            "best_layer_size": best_params['layer_size'],
            "best_lr": best_params['lr'],
            "best_weight_decay": best_params['weight_decay'],
            "best_residual_dropout": best_params['residual_dropout'],
            "best_hidden_dropout": best_params['hidden_dropout'],
            "best_n_layers": best_params['n_layers'],
            "best_d_embedding": best_params['d_embedding'],
            "best_d_hidden_factor": best_params['d_hidden_factor'],
            "best_epoch": best_params['epochs'],
            "best_cardinality": best_params['cardinality'] if args.model == 'resnext' else None
        }
        wandb.log(best_configuration)
    else:
        best_params = {}
    X_outer_preprocessed = D.build_X(
        normalization=args.normalization,
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args.cat_policy,
        seed=args.seed,
        train_idx=outer_train_idx,
        test_idx=outer_test_idx,
    )
    if D.is_regression:
        y_policy = 'mean_std'
    else:
        y_policy = None
    if not isinstance(X_outer_preprocessed, tuple):
        X_outer_preprocessed = (X_outer_preprocessed, None)
    set_random_seed(args.seed)
    Y, y_info = D.build_y(train_idx=outer_train_idx, test_idx=outer_test_idx, policy=y_policy)

    X_outer_preprocessed = tuple(None if x is None else lib.to_tensors(x) for x in X_outer_preprocessed)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X_outer_preprocessed = tuple(
            None if x is None else x.to(device) for x in X_outer_preprocessed
        )
        if isinstance(Y, dict):
            Y_device = {k: v.to(device) for k, v in Y.items()}
        else:
            Y_device = Y.to(device)
    else:
        Y_device = Y
    X_num, X_cat = X_outer_preprocessed
    if X_cat is not None:
        X_cat = X_cat.long()
    del X_outer_preprocessed
    if not D.is_multiclass:
        if isinstance(Y_device, dict):
            Y_device = {k: v.float() for k, v in Y_device.items()}
            max_index = len(Y_device)
            Y_tensor = torch.empty(max_index, device='cuda:0')
            for i, idx in enumerate(Y_device):
                Y_tensor[i] = Y_device[idx]
            Y_device = Y_tensor
        else:
            Y_device = Y_device.float()
    else:
        Y_device = Y_device.long()

    train_size = D.size(outer_train_idx)
    epoch_size = math.ceil(train_size / batch_size)
    chunk_size = None

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )

    layer_size = best_params.get('layer_size', args.layer_size)
    lr = best_params.get('lr', args.lr)
    weight_decay = best_params.get('weight_decay', args.weight_decay)
    residual_dropout = best_params.get('residual_dropout', args.residual_dropout)
    hidden_dropout = best_params.get('hidden_dropout', args.hidden_dropout)
    n_layers = best_params.get('n_layers', args.n_layers)
    d_embedding = best_params.get('d_embedding', args.d_embedding)
    d_hidden_factor = best_params.get('d_hidden_factor', args.d_hidden_factor)
    best_epoch = best_params.get('epochs', args.epochs)
    if args.model == 'resnext':
        cardinality = best_params.get('cardinality', args.cardinality)
    else:
        cardinality = None
    if args.model == 'resnet':
        model = ResNet(
            d_numerical=0 if X_num is None else X_num[outer_train_idx].shape[1],
            categories=lib.get_categories(X_cat, outer_train_idx, outer_test_idx),
            d_out=D.info['n_classes'] if D.is_multiclass else 1,
            d=layer_size,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
            d_embedding=d_embedding,
            d_hidden_factor=d_hidden_factor,
            activation='relu',
            normalization='batchnorm',
        ).to(device)
    else:
        model = ResNext(
            d_numerical=0 if X_num is None else X_num[outer_train_idx].shape[1],
            categories=lib.get_categories(X_cat, outer_train_idx, outer_test_idx),
            d_out=D.info['n_classes'] if D.is_multiclass else 1,
            d=layer_size,
            n_layers=n_layers,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
            d_embedding=d_embedding,
            d_hidden_factor=d_hidden_factor,
            activation='relu',
            normalization='batchnorm',
            cardinality=cardinality,
        ).to(device)

    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    num_parameters = lib.get_n_parameters(model)

    optimizer = lib.make_optimizer(
        "adamw",
        (
            [
                {'params': model.parameters()},
            ]
        ),
        lr,
        weight_decay,
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args.patience)
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': epoch_size,
                    'n_parameters': num_parameters,
                }.items()
            )
        )

    def apply_model(indices, idx):
        return model(
            None if X_num is None else X_num[indices][idx],
            None if X_cat is None else X_cat[indices][idx],
        )

    @torch.no_grad()
    def evaluate(indices, eval_batch_size):
        model.eval()
        metrics = {}
        predictions = {}
        while eval_batch_size:
            try:
                predictions["test"] = (
                    torch.cat(
                        [
                            apply_model(indices, idx)
                            for idx in lib.IndexLoader(
                            D.size(indices), eval_batch_size, False, device
                        )
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                eval_batch_size //= 2
                print('New eval batch size:', eval_batch_size)
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics['test'] = lib.calculate_metrics(
            D.info['task_type'],
            Y_device[indices].cpu().numpy(),  # type: ignore[code]
            predictions['test'],  # type: ignore[code]
            'logits',
            y_info,
        )
        for part_metrics in metrics.items():
            print(f'[test]', lib.make_summary(part_metrics))
        return metrics, predictions

    timer.run()
    best_test_rocauc = 0.0
    best_test_mse_loss = 1e10
    for epoch in stream.epochs(best_epoch):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model(outer_train_idx, x), Y_device[outer_train_idx][x]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')
        wandb.log({f"train_loss_outer_fold_{outer_fold}": round(sum(epoch_losses) / len(epoch_losses), 3)})

    metrics, predictions = evaluate(outer_test_idx, eval_batch_size)
    print(f"Finished outer fold {outer_fold}")
    if D.is_regression:
        print("Best mse loss:", best_test_mse_loss)
        wandb.log({f"best_test_mse_loss_outer_fold_{outer_fold}": best_test_mse_loss})
    else:
        print("Best rocauc:", best_test_rocauc)
        wandb.log({f"best_test_rocauc_outer_fold_{outer_fold}": metrics['test']['score']})


# %%
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
    wandb.log({"total_time": end_time - start_time})
    wandb.finish()
