import logging
import pathlib
import sys

import fire
import pandas

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import load_json
from ludwig.visualize import learning_curves
from ludwig.visualize import compare_performance


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def predict():
    "predict using model"
    output_directory = SCRIPT_DIR / 'output' / 'predict'
    test_file = SCRIPT_DIR / 'rotten_tomatoes_test.csv'
    experiment_name = "rt"
    model_name = "run"

    experiment_dir = experiment_name + '_' + model_name
    model_dir = SCRIPT_DIR / 'output' / 'results' / experiment_dir / 'model'

    model = LudwigModel.load(model_dir, backend='local')
    predictions, output_directory = model.predict(
        dataset=str(test_file),
        skip_save_unprocessed_output=True,
        skip_save_predictions=False,
        output_directory=output_directory,
    )


def load_training_statistics(experiment_dir):
    training_statistics = load_json(
        experiment_dir / 'training_statistics.json'
    )
    return training_statistics


def visualize_training():
    results_path = SCRIPT_DIR / 'output' / 'results'
    results_dirs = [d for d in results_path.glob('*') if d.is_dir()]
    latest_dirs = sorted(results_dirs, key=lambda f: f.name, reverse=True)
    if len(latest_dirs) == 0:
        sys.exit('Cannot find results dir in {}'.format(results_path))

    experiment_dir = latest_dirs[0]
    print('get training statistics from {}'.format(experiment_dir))
    training_statistics = load_training_statistics(experiment_dir)

    output_directory = (
        SCRIPT_DIR / 'output' / 'visualizations' / experiment_dir.name
    )
    list_of_stats = [training_statistics]

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
    output_directory = SCRIPT_DIR / 'output' / 'visualizations'

    model_name = "run"

    experiment_name1 = "rt"
    experiment_dir = experiment_name1 + '_' + model_name
    model_dir1 = SCRIPT_DIR / 'output' / 'results' / experiment_dir / 'model'

    model1 = LudwigModel.load(model_dir1, backend='local')
    eval_stats1, predictions1, output_directory1 = model1.evaluate(
        dataset=str(test_file)
    )

    experiment_name2 = "rt_zscore"
    experiment_dir = experiment_name2 + '_' + model_name
    model_dir2 = SCRIPT_DIR / 'output' / 'results' / experiment_dir / 'model'

    model2 = LudwigModel.load(model_dir2, backend='local')
    eval_stats2, predictions2, output_directory2 = model2.evaluate(
        dataset=str(test_file)
    )

    list_of_eval_stats = [eval_stats1, eval_stats2]
    model_names = [experiment_name1, experiment_name2]
    compare_performance(
        list_of_eval_stats,
        "recommended",
        model_names=model_names,
        output_directory=output_directory,
        file_format="png",
    )


def train():
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
            "train": train,
            "predict": predict,
            "visualize-training": visualize_training,
            "compare-perf": compare_perf,
        }
    )
