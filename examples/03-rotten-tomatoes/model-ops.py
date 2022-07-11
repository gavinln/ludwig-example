import logging
import pathlib
import shutil
import sys

from typing import List

import fire
import pandas

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import load_json
from ludwig.visualize import learning_curves
from ludwig.visualize import compare_performance


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def get_ludwig_output_dir():
    return pathlib.Path.home() / '.ludwig-output' / SCRIPT_DIR.name / 'output'


def move_preprocessed_input_files(
    training_set, validation_set, test_set, output_dir
):
    training_file = training_set.data_hdf5_fp
    validation_file = validation_set.data_hdf5_fp.replace(
        '.training.hdf5', '.validation.hdf5'
    )
    test_file = test_set.data_hdf5_fp.replace('.training.hdf5', '.test.hdf5')
    meta_file = training_set.data_hdf5_fp.replace(
        '.training.hdf5', '.meta.json'
    )
    # move processed input to a separate directory
    processed_input_dir = pathlib.Path(output_dir + '_preprocessed')
    processed_input_dir.mkdir()
    shutil.move(training_file, processed_input_dir)
    shutil.move(test_file, processed_input_dir)
    shutil.move(validation_file, processed_input_dir)
    shutil.move(meta_file, processed_input_dir)


def train_auto(config_file, experiment_name):
    "train model for rotten_tomatoes data set"

    df = pandas.read_csv(SCRIPT_DIR / 'rotten_tomatoes.csv')
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
    print(f'{training_set.size=}, {validation_set.size=}, {test_set.size=}')
    print('training_set_metadata keys {}'.format(training_set_metadata.keys()))
    print(f'{output_dir=}')

    # move_preprocessed_input_files(
    #     training_set, validation_set, test_set, output_dir
    # )


def train_auto_split():
    "train model with auto data split"
    train_auto(SCRIPT_DIR / 'rotten_tomatoes.yaml', 'auto-split')


def train_manual(config_file, experiment_name):
    "train model with manual data split"

    df = pandas.read_csv(SCRIPT_DIR / 'rotten_tomatoes.csv').sample(frac=1)
    output_dir = get_ludwig_output_dir() / 'results'
    print(f'data shape: {df.shape=}')
    model = LudwigModel(config=str(config_file))

    nrow = df.shape[0]
    training_set_nrow = int(nrow * 8 / 10)
    validation_set_nrow = int(nrow * 1 / 10)
    test_set_nrow = nrow - training_set_nrow - validation_set_nrow

    training_set = df.iloc[:training_set_nrow]
    validation_set = df.iloc[
        training_set_nrow : (training_set_nrow + validation_set_nrow)
    ]
    test_set = df.iloc[-test_set_nrow:]
    (training_statistics, preprocessed_data, output_dir) = model.train(
        training_set=training_set,
        validation_set=validation_set,
        # test_set=test_set,
        experiment_name=experiment_name,
        output_directory=str(output_dir),
    )
    print('training_statistics keys: {}'.format(training_statistics.keys()))
    (
        training_set,
        validation_set,
        test_set,
        training_set_metadata,
    ) = preprocessed_data
    if test_set:
        print(f'{training_set.size=}, {validation_set.size=}, {test_set.size=}')
    else:
        print(f'{training_set.size=}, {validation_set.size=}, test_set is None')
    print('training_set_metadata keys {}'.format(training_set_metadata.keys()))
    print(f'{output_dir=}')


def train_manual_split():
    "train model with manual data split"
    train_manual(SCRIPT_DIR / 'rotten_tomatoes.yaml', 'manual-split')


def predict(experiment_name):
    "predict using model"
    test_file = SCRIPT_DIR / 'rotten_tomatoes_test.csv'
    model_name = "run"

    experiment_dir = experiment_name + '_' + model_name
    output_dir = get_ludwig_output_dir() / 'predict' / experiment_dir
    model_dir = get_ludwig_output_dir() / 'results' / experiment_dir / 'model'

    model = LudwigModel.load(model_dir, backend='local')
    predictions, output_dir = model.predict(
        dataset=str(test_file),
        skip_save_unprocessed_output=True,
        skip_save_predictions=False,
        output_directory=output_dir,
    )
    print(f'{model_dir=}')
    print(f'{output_dir=}')


def predict_auto_split():
    "predict with auto data split"
    predict('auto-split')


def predict_manual_split():
    "predict with manual data split"
    predict('manual-split')


def load_training_statistics(experiment_name, model_name):
    experiment_dir = experiment_name + '_' + model_name

    training_statistics = load_json(
        get_ludwig_output_dir() / 'results' / experiment_dir
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


def visualize_manual():
    "visualize training for manual data split"
    visualize_training('manual-split')


def visualize_training_bak():
    results_path = SCRIPT_DIR / 'output' / 'results'
    results_dirs = [d for d in results_path.glob('*') if d.is_dir()]
    latest_dirs = sorted(results_dirs, key=lambda f: f.name, reverse=True)
    if len(latest_dirs) == 0:
        sys.exit('Cannot find results dir in {}'.format(results_path))

    experiment_dir = latest_dirs[0]
    print('get training statistics from {}'.format(experiment_dir))
    # training_statistics = load_training_statistics(experiment_dir)

    output_directory = (
        SCRIPT_DIR / 'output' / 'visualizations' / experiment_dir.name
    )
    # list_of_stats = [training_statistics]
    list_of_stats: List[str] = []

    list_of_models = [experiment_dir.name]
    learning_curves(
        list_of_stats,
        output_feature_name='recommended',
        model_names=list_of_models,
        output_directory=output_directory,
        file_format='png',
    )


def compare_perf():
    "compare performance of two models"
    test_file = SCRIPT_DIR / 'rotten_tomatoes_test.csv'
    output_dir = get_ludwig_output_dir()

    model_name = "run"

    experiment_name1 = "auto-split"
    experiment_dir = experiment_name1 + '_' + model_name
    model_dir1 = output_dir / 'results' / experiment_dir / 'model'

    model1 = LudwigModel.load(model_dir1, backend='local')
    eval_stats1, predictions1, output_dir1 = model1.evaluate(
        dataset=str(test_file)
    )

    experiment_name2 = "manual-split"
    experiment_dir = experiment_name2 + '_' + model_name
    model_dir2 = output_dir / 'results' / experiment_dir / 'model'

    model2 = LudwigModel.load(model_dir2, backend='local')
    eval_stats2, predictions2, output_dir2 = model2.evaluate(
        dataset=str(test_file)
    )

    list_of_eval_stats = [eval_stats1, eval_stats2]
    model_names = [experiment_name1, experiment_name2]
    compare_performance(
        list_of_eval_stats,
        "recommended",
        model_names=model_names,
        output_directory=output_dir,
        file_format="png",
    )
    print(f'{output_dir=}')


def train_bak():
    "train model"

    config_file = SCRIPT_DIR / 'rotten_tomatoes.yaml'
    experiment_name = 'rt'
    df = pandas.read_csv(SCRIPT_DIR / 'rotten_tomatoes.csv').sample(frac=1)
    output_directory = SCRIPT_DIR / 'output' / 'results'
    print(f'data shape: {df.shape=}')
    model = LudwigModel(config=str(config_file))

    nrow = df.shape[0]
    training_set_nrow = int(nrow * 8 / 10)
    validation_set_nrow = int(nrow * 1 / 10)
    test_set_nrow = nrow - training_set_nrow - validation_set_nrow

    training_set = df.iloc[:training_set_nrow]
    validation_set = df.iloc[
        training_set_nrow : (training_set_nrow + validation_set_nrow)
    ]
    test_set = df.iloc[-test_set_nrow:]
    # (training_statistics, preprocessed_data, output_directory) = model.train(
    #     dataset=df,
    #     experiment_name=experiment_name,
    #     output_directory=str(output_directory),
    # )
    (training_statistics, preprocessed_data, output_directory) = model.train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        experiment_name=experiment_name,
        output_directory=str(output_directory),
    )
    print('training_statistics keys: {}'.format(training_statistics.keys()))
    (
        training_set,
        validation_set,
        test_set,
        training_set_metadata,
    ) = preprocessed_data
    print(f'{training_set.size=}, {validation_set.size=}, {test_set.size=}')
    print('training_set_metadata keys {}'.format(training_set_metadata.keys()))
    print(f'{output_directory=}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(
        {
            "train-auto": train_auto_split,
            "train-manual": train_manual_split,
            "predict-auto": predict_auto_split,
            "predict-manual": predict_manual_split,
            "visualize-auto": visualize_auto,
            "visualize-manual": visualize_manual,
            "compare-perf": compare_perf,
        }
    )
