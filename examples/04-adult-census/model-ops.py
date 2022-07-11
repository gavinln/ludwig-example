import logging
import pathlib
import shutil
import sys

from typing import List

import fire
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import load_json
from ludwig.visualize import learning_curves
from ludwig.visualize import compare_performance

from ludwig.datasets import adult_census_income

# https://github.com/ludwig-ai/ludwig/blob/master/examples/lightgbm/train.py


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def get_dataset():
    adult_census_income.load()
    raw_dir = (
        pathlib.Path.home()
        / '.ludwig_cache'
        / 'adult_census_income_1.0'
        / 'raw'
    )
    processed_dir = (
        pathlib.Path.home()
        / '.ludwig_cache'
        / 'adult_census_income_1.0'
        / 'processed'
    )
    print(f'{raw_dir=}')
    print(f'{processed_dir=}')
    adult_census_income_df = pd.read_csv(
        processed_dir / 'adult_census_income.csv'
    )
    return adult_census_income_df


def get_ludwig_output_dir():
    return pathlib.Path.home() / '.ludwig-output' / SCRIPT_DIR.name / 'output'


def print_dataset_sizes(training_set, validation_set, test_set):
    names = ['train', 'validate', 'test']
    datasets = [training_set, validation_set, test_set]
    name_data = {
        name: dataset for name, dataset in zip(names, datasets) if dataset
    }
    name_shape = ['{}.shape={}'.format(
        name, data.to_df().shape) for name, data in name_data.items()]
    print(' '.join(name_shape))


def train_auto(config_file, experiment_name):
    "train model for adult_income data set"

    df = get_dataset()
    output_dir = get_ludwig_output_dir() / 'results'
    print(f'data shape: {df.shape=}')

    model = LudwigModel(config=str(config_file))

    (training_statistics, preprocessed_data, output_dir) = model.train(
        dataset=df,
        experiment_name=experiment_name,
        skip_save_processed_input=True,
        output_directory=str(output_dir),
    )
    print('training_statistics keys: {}'.format(training_statistics.keys()))
    (
        training_set,
        validation_set,
        test_set,
        training_set_metadata,
    ) = preprocessed_data
    print_dataset_sizes(training_set, validation_set, test_set)
    print('training_set_metadata keys {}'.format(training_set_metadata.keys()))
    print(f'{output_dir=}')


def train_auto_split():
    "train model with auto data split"
    train_auto(SCRIPT_DIR / 'adult_income.yaml', 'auto-split')


def load_training_statistics(experiment_name, model_name):
    experiment_dir = experiment_name + '_' + model_name

    training_statistics = load_json(
        get_ludwig_output_dir()
        / 'results'
        / experiment_dir
        / 'training_statistics.json'
    )
    return training_statistics


def visualize_training(experiment_name):
    model_name = "run"

    experiment_dir = experiment_name + '_' + model_name
    training_statistics = load_training_statistics(experiment_name, model_name)

    output_dir = get_ludwig_output_dir() / 'viz' / experiment_dir
    list_of_stats = [training_statistics]

    list_of_models = [experiment_dir]
    learning_curves(
        list_of_stats,
        output_feature_name='recommended',
        model_names=list_of_models,
        output_directory=output_dir,
        file_format='png',
    )
    print(f'{output_dir=}')


def visualize_auto():
    "visualize training for auto data split"
    visualize_training('auto-split')


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(
        {
            "get-dataset": get_dataset,
            "train-auto": train_auto_split,
            "visualize-auto": visualize_auto
        }
    )
