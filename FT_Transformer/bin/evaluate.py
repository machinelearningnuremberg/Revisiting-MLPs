import argparse
import math
import signal
from contextlib import contextmanager

import optuna
import zero
import torch.nn as nn
import torch
import torch.nn.functional as F
import lib
import wandb

from sklearn.model_selection import StratifiedKFold
from ft_transformer import Transformer

# Create the parser
parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

# Add the arguments
parser.add_argument('--batch_size', type=int, default=512,
                    help='The batch size for training. Default is 32.')
parser.add_argument('--experiment_name', type=str, default='FT_optuna_original',
                    help='The name of the experiment. Default is "test".')
parser.add_argument('--eval_batch_size', type=int, default=8192,
                    help='The batch size for evaluation. Default is 8192.')
parser.add_argument('--dataset', type=int, default=1478,
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
parser.add_argument('--outer_fold', type=int, default=8, help='The outer fold to use. Default is 0')
parser.add_argument('--n_trials', type=int, default=100,
                    help='The number of trials to use for HPO. Default is 100')
args = parser.parse_args()


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
    zero.set_randomness(args.seed)
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
    best_params = {
        "n_layers": 5,
        "d_token": 440,
        "residual_dropout": 0.1430378732744839,
        "attn_dropout": 0.30138168803582194,
        "ffn_dropout": 0.27244159149844843,
        "d_ffn_factor": 1.5139762653444762,
        "lr": 0.00019578897201212985,
        "weight_decay": 0.00002054762512591135,
    }
    n_layers = best_params['n_layers']
    d_token = best_params['d_token']
    # Adjust d_token to the nearest lower number divisible by n_heads
    d_token = d_token * args.n_heads
    residual_dropout = best_params['residual_dropout']
    attn_dropout = best_params['attn_dropout']
    ffn_dropout = best_params['ffn_dropout']
    d_ffn_factor = best_params['d_ffn_factor']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    best_configuration = {
        "best_n_layers": n_layers,
        "best_d_token": d_token,
        "best_residual_dropout": residual_dropout,
        "best_attn_dropout": attn_dropout,
        "best_ffn_dropout": ffn_dropout,
        "best_d_ffn_factor": d_ffn_factor,
        "best_lr": lr,
        "best_weight_decay": weight_decay,
    }
    wandb.log(best_configuration)
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
    zero.set_randomness(args.seed)
    timer = zero.Timer()
    timer.run()
    D = lib.Dataset.from_openml(args.dataset)
    run_name = f"{D.info['dataset_name']}_outerFold_{args.outer_fold}"
    wandb.init(project=args.experiment_name,
               name=run_name,
               id="lrt71wme",
               resume="must",
               config=args)
    outer_kfold = StratifiedKFold(n_splits=10, shuffle=True)
    outer_folds = list(outer_kfold.split(D.X, D.y))
    run_single_outer_fold(args.outer_fold, D, outer_folds)
