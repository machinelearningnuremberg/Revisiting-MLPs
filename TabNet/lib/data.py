import dataclasses as dc
import os
import pickle
import random

import pandas as pd
import typing as ty
import warnings
import openml
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from . import env, util

ArrayDict = ty.Dict[str, np.ndarray]


def normalize(
        X: ArrayDict, normalization: str, train_idx: ty.List, seed: int, noise: float = 1e-3
) -> ArrayDict:
    X_train = X[train_idx].copy()
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        if noise:
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
                X_train.shape
            )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    return normalizer.transform(X)  # type: ignore[code]


@dc.dataclass
class Dataset:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    X: ty.Optional[ArrayDict]
    info: ty.Dict[str, ty.Any]

    @classmethod
    def from_openml(cls, dataset_id: int) -> 'Dataset':
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )

        # Separate numerical and categorical features
        N = X.select_dtypes(include=[np.number]).values
        C = X.select_dtypes(exclude=[np.number]).values
        N = N if N.shape[1] > 0 else None
        C = C if C.shape[1] > 0 else None

        if dataset_id in [43466, 1099, 42184, 43611, 560, 44025, 703, 194, 42727, 566, 43939, 497, 524]:
            task_type = 'regression'
        else:
            task_type = 'binclass' if len(y.unique()) == 2 else 'multiclass'

        # Package the data into ArrayDicts
        data = {
            'N': N,
            'C': C,
            'y': y,
            'X': X,
        }
        # Create the info dictionary
        info = {
            'n_classes': len(y.unique()),
            'task_type': task_type,
            'n_num_features': N.shape[1] if N is not None else 0,
            'n_cat_features': C.shape[1] if C is not None else 0,
            'dataset_name': dataset.name,
        }
        dataset_obj = cls(N=data['N'], C=data['C'], y=data['y'], X=data['X'], info=info)

        return dataset_obj

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == util.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == util.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == util.REGRESSION

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_cat_features(self) -> int:
        return self.info['n_cat_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, indices: ty.List) -> int:
        X = self.N if self.N is not None else self.C
        assert X is not None
        return len(X[indices])

    def build_X(
            self,
            train_idx: ty.List[int],
            test_idx: ty.List[int],
            X=None,
            *,
            normalization: ty.Optional[str],
            num_nan_policy: str,
            cat_nan_policy: str,
            cat_policy: str,
            cat_min_frequency: float = 0.0,
            seed: int,
    ) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:
        N = None
        C = None
        if X is not None:
            N = X.select_dtypes(include=[np.number]).values
            C = X.select_dtypes(exclude=[np.number]).values
            N = N if N.shape[1] > 0 else None
            C = C if C.shape[1] > 0 else None
        else:
            if self.N is not None:
                N = deepcopy(self.N)
            if self.C is not None:
                C = deepcopy(self.C)

        if N is not None:
            N = N.astype(np.float32)
            for idx in [train_idx, test_idx]:
                num_nan_masks = np.isnan(N[idx])
                if np.any(num_nan_masks):
                    if num_nan_policy == 'mean':
                        num_new_values = np.nanmean(N[train_idx], axis=0)
                    else:
                        util.raise_unknown('numerical NaN policy', num_nan_policy)
                    replaced_values = np.where(num_nan_masks, num_new_values[None, :], N[idx])
                    N[idx] = replaced_values

            if normalization:
                N = normalize(N, normalization, train_idx, seed)

        else:
            N = None

        if cat_policy == 'drop' or self.C is None or C is None:
            assert N is not None
            return N, None

        # Handle missing values
        cat_nan_masks_train = np.vectorize(lambda x: x == 'nan' or isinstance(x, float) and np.isnan(x))(C[train_idx])
        cat_nan_masks_test = np.vectorize(lambda x: x == 'nan' or isinstance(x, float) and np.isnan(x))(C[test_idx])
        if cat_nan_policy == 'new':
            cat_new_value = '___null___'
            C[train_idx] = np.where(cat_nan_masks_train, cat_new_value, C[train_idx])
            C[test_idx] = np.where(cat_nan_masks_test, cat_new_value, C[test_idx])
        elif cat_nan_policy == 'most_frequent':
            imputer = SimpleImputer(strategy=cat_nan_policy)
            imputer.fit(C[train_idx])
            C = imputer.transform(C)
        else:
            util.raise_unknown('categorical NaN policy', cat_nan_policy)

        if cat_min_frequency:
            min_count = round(len(train_idx) * cat_min_frequency)
            rare_value = '___rare___'
            C_new_train, C_new_test = [], []

            for column_idx in range(C.shape[1]):
                counter = Counter(C[train_idx, column_idx])
                popular_categories = {k for k, v in counter.items() if v >= min_count}

                C_new_train.append(
                    [
                        (x if x in popular_categories else rare_value)
                        for x in C[train_idx, column_idx]
                    ]
                )

                C_new_test.append(
                    [
                        (x if x in popular_categories else rare_value)
                        for x in C[test_idx, column_idx]
                    ]
                )

            # Reshape the new categorical data
            C_train = np.array(C_new_train).T
            C_test = np.array(C_new_test).T

            # Update C with the new categorical data
            C[train_idx] = C_train
            C[test_idx] = C_test

        unknown_value = np.iinfo('int64').max - 3
        encoder = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown_value,
            dtype=np.int64,
        ).fit(C[train_idx])

        C = encoder.transform(C)

        max_values = C[train_idx].max(axis=0)
        for column_idx in range(C[test_idx].shape[1]):
            C[test_idx][C[test_idx][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
            )

        if cat_policy == 'indices':
            result = (N, C)
        elif cat_policy == 'ohe':
            ohe = sklearn.preprocessing.OneHotEncoder(
                handle_unknown='ignore', sparse=False, dtype='float32'
            )
            ohe.fit(C[train_idx])
            C = ohe.transform(C)
            result = C if N is None else np.hstack((N, C))
        elif cat_policy == 'counter':
            assert seed is not None
            loo = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
            loo.fit(C[train_idx], self.y[train_idx])
            C = loo.transform(C).astype('float32')
            if normalization:
                C = normalize({'data': C}, normalization, train_idx, seed, inplace=True)['data']
            result = C if N is None else np.hstack((N, C))
        else:
            util.raise_unknown('categorical policy', cat_policy)
        return result

    def build_y(
            self, train_idx: ty.List[int], test_idx, policy: ty.Optional[str] = None, y=None
    ) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
        if self.is_regression:
            assert policy == 'mean_std'
        if y is None:
            y = deepcopy(self.y)
        if policy:
            if not self.is_regression:
                warnings.warn('y_policy is not None, but the task is NOT regression')
                info = None
            elif policy == 'mean_std':
                mean, std = self.y[train_idx].mean(), self.y[train_idx].std()
                y = {k: (v - mean) / std for k, v in y.items()}
                info = {'policy': policy, 'mean': mean, 'std': std}
            else:
                util.raise_unknown('y policy', policy)
        else:
            info = None
            label_encoder = LabelEncoder().fit(y.iloc[train_idx])
            y = label_encoder.transform(y)
        return y, info


def to_tensors(data: np.ndarray) -> torch.Tensor:
    if isinstance(data, dict):
        return {k: torch.as_tensor(v) for k, v in data.items()}
    return torch.as_tensor(data)


def load_dataset_info(dataset_name: str) -> ty.Dict[str, ty.Any]:
    info = util.load_json(env.DATA_DIR / dataset_name / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    return info


def set_random_seed(seed):
    """
    Set the seed for random number generation in Python, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
