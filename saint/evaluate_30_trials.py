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
import numpy as np
from lib.data import set_random_seed

from augmentations import embed_data_mask
from models import SAINT
from sklearn.model_selection import StratifiedKFold

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--batch_size', type=int, default=512,
                    help='The batch size for training. Default is 32.')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--eval_batch_size', type=int, default=8192,
                    help='The batch size for evaluation. Default is 8192.')
parser.add_argument('--dataset', type=int, default=1510,
                    help='The dataset ID to use. Default is 45068 (adult).')
parser.add_argument('--seed', type=int, default=0,
                    help='The random seed for reproducibility. Default is 42.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model for. Default is 100.')
parser.add_argument('--patience', type=int, default=16,
                    help='Number of epochs to wait for improvement before early stopping. Default is 16.')
parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str,
                    choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weight_decay', type=float, default=1e-05,
                    help='The weight decay to use. Default is 1e-5.')
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
    history = target_run.scan_history(keys=['avg_test_rocauc'])
    for i, row in enumerate(history):
        if i >= num_trials:
            break
        if 'avg_test_rocauc' in row and row['avg_test_rocauc'] > best_rocauc:
            best_rocauc = row['avg_test_rocauc']
            best_step = i

    if best_step is None:
        raise ValueError("Best rocauc not found within the first 30 trials")

    # Second scan for the HPs at the best step
    hp_keys = ['embedding_size', 'transformer_depth', 'attention_dropout', 'ff_dropout', 'lr', 'weight_decay']
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
    # batch size settings for datasets in (Grinsztajn et al., 2022)
    if D.n_features <= 32:
        batch_size = 512
        eval_batch_size = 8192
    elif D.n_features <= 100:
        batch_size = 64
        eval_batch_size = 256
    elif D.n_features <= 1000:
        batch_size = 32
        eval_batch_size = 64
    else:
        batch_size = 16
        eval_batch_size = 16

    best_params = load_best_config('t4tab/SAINT_optuna', D.info['dataset_name'], args.outer_fold)

    X_outer_preprocessed = D.build_X(
        normalization=args.normalization,
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args.cat_policy,
        seed=args.seed,
        train_idx=outer_train_idx,
        test_idx=outer_test_idx,
    )
    if not isinstance(X_outer_preprocessed, tuple):
        X_outer_preprocessed = (X_outer_preprocessed, None)
    set_random_seed(args.seed)
    Y, y_info = D.build_y(train_idx=outer_train_idx, test_idx=outer_test_idx)

    X_outer_preprocessed = tuple(None if x is None else lib.to_tensors(x) for x in X_outer_preprocessed)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X_outer_preprocessed = tuple(
            None if x is None else x.to(device) for x in X_outer_preprocessed
        )
        Y_device = Y.to(device)
    else:
        Y_device = Y
    X_num, X_cat = X_outer_preprocessed
    if X_cat is not None:
        X_cat = X_cat.long()
    del X_outer_preprocessed
    if not D.is_multiclass:
        Y_device = Y_device.float()

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
    embedding_size = best_params['embedding_size']
    embedding_size = embedding_size if D.n_features < 50 else 8
    transformer_depth = best_params['transformer_depth']
    attention_dropout = best_params['attention_dropout']
    ff_dropout = best_params['ff_dropout']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']

    cat_dims = lib.get_categories(X_cat, outer_train_idx, outer_test_idx)
    if cat_dims is not None:
        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)
    model = SAINT(
        categories=tuple(cat_dims) if cat_dims is not None else None,
        num_continuous=X_num[outer_train_idx].shape[1] if X_num is not None else 0,
        dim=embedding_size,
        dim_out=1,
        depth=transformer_depth,
        heads=args.attention_heads,
        attn_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=args.cont_embeddings,
        attentiontype=args.attentiontype,
        final_mlp_style=args.final_mlp_style,
        y_dim=D.info['n_classes'] if D.is_multiclass else 1,
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    num_parameters = lib.get_n_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        x_num = None if X_num is None else X_num[indices][idx]
        x_cat = None if X_cat is None else X_cat[indices][idx]
        if x_cat is not None:
            zero_column = torch.zeros(x_cat.size(0), 1).to(device)
            x_cat = torch.cat((x_cat, zero_column), dim=1).long().to(device)
        num_mask = torch.ones_like(x_num).long() if x_num is not None else None
        cat_mask = torch.ones_like(x_cat).long() if x_cat is not None else None
        _, x_cat_enc, x_num_enc = embed_data_mask(x_cat, x_num, cat_mask, num_mask, model, False)
        reps = model.transformer(x_cat_enc, x_num_enc)
        y_reps = reps[:, 0, :]
        y_outs = model.mlpfory(y_reps)
        return y_outs.squeeze(1)

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
            Y[indices].numpy(),  # type: ignore[code]
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
    for epoch in stream.epochs(args.epochs):
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
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics['test']['score'])

        if progress.success:
            print('New best epoch!')
            best_epoch = stream.epoch
            if metrics['test']['score'] > best_test_rocauc:
                best_test_rocauc = metrics['test']['score']

        elif progress.fail:
            break
    print(f"Finished outer fold {outer_fold}")
    wandb.log({f"best_test_rocauc_outer_fold_{outer_fold}": best_test_rocauc})
    wandb.finish()


# %%
if __name__ == "__main__":
    # %%
    set_random_seed(args.seed)
    timer = zero.Timer()
    timer.run()
    D = lib.Dataset.from_openml(args.dataset)
    run_name = f"{D.info['dataset_name']}_outerFold_{args.outer_fold}"
    wandb.init(project=args.experiment_name,
               name=run_name,
               config=args)
    outer_kfold = StratifiedKFold(n_splits=10, shuffle=True)
    outer_folds = list(outer_kfold.split(D.X, D.y))
    run_single_outer_fold(args.outer_fold, D, outer_folds)