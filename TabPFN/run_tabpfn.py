import argparse
import json
import os

import openml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier
from utils import set_random_seed
import torch
import numpy as np
import pandas as pd


def main(args: argparse.Namespace):

    seed = args.seed
    set_random_seed(seed)
    outer_fold = args.outer_fold
    dataset_id = args.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id, download_data=False)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )

    categorical_column_names = X.columns[categorical_indicator]
    X = pd.get_dummies(X, columns=categorical_column_names)

    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    train_idx, test_idx = splits[outer_fold]
    nr_classes = len(np.unique(y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier = TabPFNClassifier(device=device, seed=seed, N_ensemble_configurations=32)

    classifier.fit(X_train, y_train)
    p_eval = classifier.predict_proba(X_test)
    y_eval = classifier.predict(X_test)
    if nr_classes == 2:
        p_eval = p_eval[:, 1]

    auroc_test_value = roc_auc_score(y_test, p_eval, multi_class='ovo')

    acc_test_value = accuracy_score(y_test, y_eval)

    result_path = os.path.join(
        args.output_dir,
        'tabpfn',
        f'{dataset_id}',
        f'{outer_fold}',
    )

    os.makedirs(result_path, exist_ok=True)
    result_dict = {
        'test_auroc': auroc_test_value,
        'test_acc': acc_test_value,
    }

    with open(os.path.join(result_path, 'result.json'), 'w') as f:
        json.dump(result_dict, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    parser.add_argument(
        '--outer_fold',
        type=int,
        default=2,
        help='Outer fold iteration.',
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=31,
        help='Dataset id',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the results',
    )

    args = parser.parse_args()

    main(args)