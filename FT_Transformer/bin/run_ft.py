import argparse
import math
import time

import optuna
import zero
import torch.nn as nn
import torch
import torch.nn.functional as F
import lib
import wandb

from sklearn.model_selection import StratifiedKFold, KFold
from ft_transformer import Transformer

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--batch_size', type=int, default=512,
                    help='The batch size for training. Default is 32.')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--eval_batch_size', type=int, default=8192,
                    help='The batch size for evaluation. Default is 8192.')
parser.add_argument('--dataset', type=int, default=44025,
                    help='The dataset ID to use. Default is 45068 (adult).')
parser.add_argument('--seed', type=int, default=0,
                    help='The random seed for reproducibility. Default is 42.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model for. Default is 100.')
parser.add_argument('--patience', type=int, default=16,
                    help='Number of epochs to wait for improvement before early stopping. Default is 16.')
parser.add_argument('--activation', type=str, default='reglu', choices=['relu', 'geglu', 'sigmoid'],
                    help='The activation function to use. Default is "reglu".')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='The learning rate to use. Default is 1e-4.')
parser.add_argument('--weight_decay', type=float, default=1e-05,
                    help='The weight decay to use. Default is 1e-5.')
parser.add_argument('--attn_dropout', type=float, default=0.2,
                    help='The dropout to use for the attention layers. Default is 0.2.')
parser.add_argument('--d_ffn_factor', type=float, default=1.333333333333333)
parser.add_argument('--d_token', type=int, default=192,
                    help='The dimensionality of the token embeddings. Default is 192.')
parser.add_argument('--ffn_dropout', type=float, default=0.1,
                    help='The dropout to use for the feed-forward layers. Default is 0.1.')
parser.add_argument('--n_heads', type=int, default=8,
                    help='The number of heads to use for the attention layers. Default is 8.')
parser.add_argument('--n_layers', type=int, default=3,
                    help='The number of layers to use for the transformer. Default is 3.')
parser.add_argument('--prenormalization', type=bool, default=True,
                    help='Whether to use pre-normalization. Default is True.')
parser.add_argument('--residual_dropout', type=float, default=0.0,
                    help='The dropout to use for the residual connections. Default is 0.0.')
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


def objective(trial, X_outer, Y_outer, batch_size, eval_batch_size, start_time, time_limit, n_inner_folds=9):
    # Search space
    n_layers = trial.suggest_int('n_layers', 1, 6)
    d_token = trial.suggest_int('d_token', 8, 64)
    d_token = d_token * args.n_heads
    residual_dropout = trial.suggest_float('residual_dropout', 0.0, 0.2)
    attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.5)
    ffn_dropout = trial.suggest_float('ffn_dropout', 0.0, 0.5)
    d_ffn_factor = trial.suggest_float('d_ffn_factor', 2 / 3, 8 / 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 10, 500)

    hyperparameters = {
        "n_layers": n_layers,
        "d_token": d_token,
        "residual_dropout": residual_dropout,
        "attn_dropout": attn_dropout,
        "ffn_dropout": ffn_dropout,
        "d_ffn_factor": d_ffn_factor,
        "lr": lr,
        "weight_decay": weight_decay,
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
        zero.set_randomness(args.seed)
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
        model_dict = {
            "activation": args.activation,
            "attention_dropout": attn_dropout,
            "d_ffn_factor": d_ffn_factor,
            "d_token": d_token,
            "ffn_dropout": ffn_dropout,
            "initialization": "kaiming",
            "n_heads": args.n_heads,
            "n_layers": n_layers,
            "prenormalization": args.prenormalization,
            "residual_dropout": residual_dropout,
        }
        model = Transformer(
            d_numerical=0 if X_num is None else X_num[train_idx].shape[1],
            categories=lib.get_categories(X_cat, train_idx, test_idx),
            d_out=D.info['n_classes'] if D.is_multiclass else 1,
            **model_dict,
        ).to(device)
        if torch.cuda.device_count() > 1:  # type: ignore[code]
            print('Using nn.DataParallel')
            model = nn.DataParallel(model)
        num_parameters = lib.get_n_parameters(model)

        def needs_wd(name):
            return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

        for x in ['tokenizer', '.norm', '.bias']:
            assert any(x in a for a in (b[0] for b in model.named_parameters()))
        parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
        optimizer = lib.make_optimizer(
            "adamw",
            (
                [
                    {'params': parameters_with_wd},
                    {'params': parameters_without_wd, 'weight_decay': 0.0},
                ]
            ),
            lr,
            weight_decay,
        )

        stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
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
                                        eval_batch_size,
                                        start_time, time_limit),
                n_trials=args.n_trials)
        except optuna.exceptions.OptunaError as e:
            print(f"Optimization stopped: {e}")
        best_params = study.best_params
        best_configuration = {
            "best_n_layers": best_params['n_layers'],
            "best_d_token": best_params['d_token'],
            "best_residual_dropout": best_params['residual_dropout'],
            "best_attn_dropout": best_params['attn_dropout'],
            "best_ffn_dropout": best_params['ffn_dropout'],
            "best_d_ffn_factor": best_params['d_ffn_factor'],
            "best_lr": best_params['lr'],
            "best_weight_decay": best_params['weight_decay'],
            "best_epoch": best_params['epochs']
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
    zero.set_randomness(args.seed)
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
    n_layers = best_params.get('n_layers', 3)
    d_token = best_params.get('d_token', 24)
    # Adjust d_token to the nearest lower number divisible by n_heads
    d_token = d_token * args.n_heads
    residual_dropout = best_params.get('residual_dropout', 0.0)
    attn_dropout = best_params.get('attn_dropout', 0.2)
    ffn_dropout = best_params.get('ffn_dropout', 0.1)
    d_ffn_factor = best_params.get('d_ffn_factor', 1.333333333333333)
    lr = best_params.get('lr', 1e-4)
    weight_decay = best_params.get('weight_decay', 1e-5)
    epochs = best_params.get('epochs', 100)
    model_dict = {
        "activation": args.activation,
        "attention_dropout": attn_dropout,
        "d_ffn_factor": d_ffn_factor,
        "d_token": d_token,
        "ffn_dropout": ffn_dropout,
        "initialization": "kaiming",
        "n_heads": args.n_heads,
        "n_layers": n_layers,
        "prenormalization": args.prenormalization,
        "residual_dropout": residual_dropout,
    }
    model = Transformer(
        d_numerical=0 if X_num is None else X_num[outer_train_idx].shape[1],
        categories=lib.get_categories(X_cat, outer_train_idx, outer_test_idx),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **model_dict,
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    num_parameters = lib.get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    for x in ['tokenizer', '.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = lib.make_optimizer(
        "adamw",
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        lr,
        weight_decay,
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
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
    best_test_rocauc = 0.0
    best_test_mse_loss = 1e10
    for epoch in stream.epochs(epochs):
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
    wandb.finish()


# %%
if __name__ == "__main__":
    # %%
    zero.set_randomness(args.seed)
    timer = zero.Timer()
    timer.run()
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
