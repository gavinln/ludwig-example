import logging
import pathlib
import shutil

from ludwig.api import LudwigModel
import pandas


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


def main():
    df = pandas.read_csv(SCRIPT_DIR / 'rotten_tomatoes.csv')
    config_file = SCRIPT_DIR / 'rotten_tomatoes.yaml'
    output_directory = SCRIPT_DIR / 'results'
    print(f'data shape: {df.shape=}')
    model = LudwigModel(config=str(config_file))

    # experiment_name = "api_experiment", model_name = "run"
    (training_statistics, preprocessed_data, output_directory) = model.train(
        dataset=df,
        skip_save_processed_input=False,
        output_directory=str(output_directory)
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

    move_preprocessed_input_files(
        training_set, validation_set, test_set, output_directory
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
