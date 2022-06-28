import logging
import pathlib
import shutil

import fire
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.datasets import higgs

from ludwig.utils.data_utils import load_json
from ludwig.visualize import learning_curves
from ludwig.visualize import compare_performance


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def move_preprocessed_input_files(
    training_set, validation_set, test_set, output_directory
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
    processed_input_dir = pathlib.Path(output_directory + '_preprocessed')
    processed_input_dir.mkdir()
    shutil.move(training_file, processed_input_dir)
    shutil.move(test_file, processed_input_dir)
    shutil.move(validation_file, processed_input_dir)
    shutil.move(meta_file, processed_input_dir)


def train(config_file, experiment_name):
    "train model"
    output_directory = SCRIPT_DIR / 'output' / 'results'
    model = LudwigModel(config=str(config_file))
    higgs_df = higgs.load()
    higgs_data_file = SCRIPT_DIR / "higgs_small.parquet"
    # higgs_data_file = (
    #     "s3://data-science.s3.liftoff.io/datascience/temp/higgs_small.parquet"
    # )
    higgs_df.to_parquet(
        higgs_data_file, engine="pyarrow", partition_cols="label"
    )

    (training_statistics, preprocessed_data, output_directory) = model.train(
        dataset=higgs_data_file,
        data_format="parquet",
        experiment_name=experiment_name,
        skip_save_processed_input=False,
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

    # move_preprocessed_input_files(
    #     training_set, validation_set, test_set, output_directory
    # )


def train_rt():
    "train model with rotten_tomatoes.yaml"
    assert False
    train(SCRIPT_DIR / 'rotten_tomatoes.yaml', 'rt')


def train_higgs_small():
    "train model with higgs small dataset"
    train(SCRIPT_DIR / 'small_config.yaml', 'higgs_small')


def train_rt_zscore():
    "train model with rotten_tomatoes_zscore.yaml"
    assert False
    train(SCRIPT_DIR / 'rotten_tomatoes_zscore.yaml', 'rt_zscore')


def predict():
    "predict using model"
    assert False
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


def load_training_statistics(experiment_name, model_name):
    assert False
    experiment_dir = experiment_name + '_' + model_name

    training_statistics = load_json(
        SCRIPT_DIR
        / 'output'
        / 'results'
        / experiment_dir
        / 'training_statistics.json'
    )
    return training_statistics


def visualize_training():
    assert False
    experiment_name = 'rt'
    model_name = "run"

    experiment_dir = experiment_name + '_' + model_name
    training_statistics = load_training_statistics(experiment_name, model_name)

    output_directory = SCRIPT_DIR / 'output' / 'visualizations'
    list_of_stats = [training_statistics]

    list_of_models = [experiment_dir]
    learning_curves(
        list_of_stats,
        output_feature_name='recommended',
        model_names=list_of_models,
        output_directory=output_directory,
        file_format='png',
    )


def compare_perf():
    "compare performance of two models"
    assert False
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


def test_s3_read_write():
    df = pd.DataFrame({"a": [0, 0, 1, 1], "b": [1, 2, 1, 2]})
    print(df)
    s3_file = 's3://data-science.s3.liftoff.io/datascience/temp/example.csv'
    df.to_csv(s3_file)
    df2 = pd.read_csv(s3_file)
    print(df2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # ray.init(address="127.0.0.1:6380")
    fire.Fire(
        {
            "train-higgs-small": train_higgs_small,
            # "train-rt-zscore": train_rt_zscore,
            # "predict": predict,
            # "visualize-training": visualize_training,
            # "compare-perf": compare_perf,
        }
    )
