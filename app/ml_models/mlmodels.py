from pathlib import Path
from pickle import load, dump
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from time import asctime


class MLModelsDAO:
    def __init__(self) -> None:
        """
        Initializes MLModelsDAO class instance.

        This function initializes MLModelsDAO class instance. If serialized ``models_info`` dictionary exists in
        the storage folder then class attributes are initialized from this dictionary.
        """
        self.available_classes = ['Random Forest Classifier', 'LightGBM Classifier']
        self.available_params = {
            'Random Forest Classifier': [
                'n_estimators',
                'criterion',
                'max_depth',
                'min_samples_split',
                'min_samples_leaf',
                'min_weight_fraction_leaf',
                'max_features',
                'max_leaf_nodes',
                'min_impurity_decrease',
                'bootstrap',
                'max_samples'
            ],
            'LightGBM Classifier': [
                'boosting_type',
                'num_leaves',
                'max_depth',
                'learning_rate',
                'num_iterations',
                'subsample_for_bin',
                'objective',
                'min_split_gain',
                'min_child_samples',
                'subsample',
                'subsample_freq',
                'colsample_bytree',
                'reg_alpha',
                'reg_lambda'
            ]
        }
        api_storage_dir = Path('D:/Projects/pycharm/restful_api/storage')
        self.models_info_path = api_storage_dir.joinpath('models_info.pkl')
        self.models_storage_dir = api_storage_dir.joinpath('models')

        api_storage_dir.mkdir(exist_ok=True)
        self.models_storage_dir.mkdir(exist_ok=True)
        if self.models_storage_dir.exists() and self.models_info_path.exists():
            with self.models_info_path.open(mode='rb') as file:
                self.models_info = load(file)
            self.counter = max(self.models_info.keys()) + 1
        else:
            self.models_info = dict()
            self.counter = 0

    def get_model_path(self, id: int) -> str:
        """Returns the path where the model `id` is stored.

        Args:
            id (int): The model ID.

        Returns:
            str: The path where requested model is stored.

        Raises:
            KeyError: If model `id` entry does not exists in the `models_info` dictionary.
        """
        if id not in self.models_info.keys():
            raise KeyError(f'ml_model {id} doesnt exist')
        return self.models_info[id]['model path']

    def create(self, class_: str, X: dict, y: dict, params: dict) -> dict:
        """Trains the model.

        This function initializes the model of `class_` with parameters provided in `params`. Then
        this model instance is trained on features `X` and target `y`.

        Args:
            class_ (str): Model class.
            X (dict): Features in dictionary format (as retrieved by ``pandas.DataFrame.to_dict()``).
            y (dict): Target in dictionary format (as retrieved by ``pandas.Series.to_dict()``).
            params (dict): Dictionary with model parameters.

        Returns:
            dict: Dictionary with information about model class, parameters, file path and last train time.
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)
        if class_ == 'Random Forest Classifier':
            ml_model = RandomForestClassifier(**params)
        elif class_ == 'LightGBM Classifier':
            ml_model = LGBMClassifier(**params)
        ml_model.fit(X, y)
        model_path = self.models_storage_dir.joinpath(f'{self.counter}.pkl')
        with model_path.open(mode='wb') as file:
            dump(ml_model, file)
        self.models_info[self.counter] = {
            'class': class_,
            'params': ml_model.get_params(),
            'model path': str(model_path),
            'train time': asctime()
        }
        with self.models_info_path.open(mode='wb') as file:
            dump(self.models_info, file)
        self.counter += 1
        return self.models_info[self.counter - 1]

    def update(self, id: int, X: dict, y: dict) -> dict:
        """Retrain the model `id` with on features `X` and target `y`.

        Args:
            id (int): The model ID.
            X (dict): Features in dictionary format (as retrieved by ``pandas.DataFrame.to_dict()``).
            y (dict): Target in dictionary format (as retrieved by ``pandas.Series.to_dict()``).

        Returns:
            dict: Dictionary with information about model class, parameters, file path and last train time.

        Raises:
            KeyError: If model `id` entry does not exists in the `models_info` dictionary.
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)
        model_path = self.get_model_path(id)
        with open(model_path, 'rb') as file:
            model = load(file)
        model.fit(X, y)
        with open(model_path, 'wb') as file:
            dump(model, file)
        self.models_info[id]['train time'] = asctime()
        with open(self.models_info_path, 'wb') as file:
            dump(self.models_info, file)
        return self.models_info[id]

    def predict(self, id: int, X: dict) -> dict:
        """Returns the prediction of the model `id`.

        This function makes the prediction with  model `id` based on features from `X`.

        Args:
            id (int): The model ID.
            X (dict):  Features in dictionary format (as retrieved by ``pandas.DataFrame.to_dict()``).

        Returns:
            dict: Dictionary where object indexes are keys and predictions are values.

        Raises:
            KeyError: If model `id` entry does not exists in the `models_info` dictionary.
        """
        X = pd.DataFrame(X)
        model_path = self.get_model_path(id)
        with open(model_path, 'rb') as file:
            model = load(file)
        return pd.Series(model.predict(X)).to_dict()


    def delete(self, id: int) -> None:
        """Delete the model `id`.

        Args:
            id (int):  The model ID.

        Raises:
            KeyError: If model `id` entry does not exists in the `models_info` dictionary.
        """
        model_path = self.get_model_path(id)
        Path(model_path).unlink()
        del self.models_info[id]
        with open(self.models_info_path, 'wb') as file:
            dump(self.models_info, file)