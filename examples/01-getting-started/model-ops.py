import logging
import pathlib
import shutil
import sys

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


def train(config_file, experiment_name):
    "train model"

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


def train_rt():
    "train model with rotten_tomatoes.yaml"
    train(SCRIPT_DIR / 'rotten_tomatoes.yaml', 'rt')


def train_rt_zscore():
    "train model with rotten_tomatoes_zscore.yaml"
    train(SCRIPT_DIR / 'rotten_tomatoes_zscore.yaml', 'rt_zscore')


def predict():
    "predict using model"
    output_dir = get_ludwig_output_dir() / 'predict'
    test_file = SCRIPT_DIR / 'rotten_tomatoes_test.csv'
    experiment_name = "rt"
    model_name = "run"

    experiment_dir = experiment_name + '_' + model_name
    model_dir = output_dir / 'results' / experiment_dir / 'model'

    model = LudwigModel.load(model_dir, backend='local')
    predictions, output_dir = model.predict(
        dataset=str(test_file),
        skip_save_unprocessed_output=True,
        skip_save_predictions=False,
        output_directory=output_dir,
    )
    print(f'{model_dir=}')
    print(f'{output_dir=}')


def load_training_statistics(experiment_name, model_name):
    experiment_dir = experiment_name + '_' + model_name

    training_statistics = load_json(
        get_ludwig_output_dir() / 'results' / experiment_dir
        / 'training_statistics.json'
    )
    return training_statistics


def visualize_training():
    experiment_name = 'rt'
    model_name = "run"

    experiment_dir = experiment_name + '_' + model_name
    training_statistics = load_training_statistics(experiment_name, model_name)

    output_dir = get_ludwig_output_dir() / 'visualizations'
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


def compare_perf():
    "compare performance of two models"
    test_file = SCRIPT_DIR / 'rotten_tomatoes_test.csv'
    output_dir = get_ludwig_output_dir()

    model_name = "run"

    experiment_name1 = "rt"
    experiment_dir = experiment_name1 + '_' + model_name
    model_dir1 = output_dir / 'results' / experiment_dir / 'model'

    model1 = LudwigModel.load(model_dir1, backend='local')
    eval_stats1, predictions1, output_dir1 = model1.evaluate(
        dataset=str(test_file)
    )

    experiment_name2 = "rt_zscore"
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(
        {
            "train-rt": train_rt,
            "train-rt-zscore": train_rt_zscore,
            "predict": predict,
            "visualize-training": visualize_training,
            "compare-perf": compare_perf,
        }
    )
