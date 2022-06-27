import logging
import pathlib

from ludwig.api import LudwigModel


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    output_directory = SCRIPT_DIR / 'predict'
    test_file = SCRIPT_DIR / 'rotten_tomatoes_test.csv'
    model_dir = SCRIPT_DIR / 'results' / 'api_experiment_run' / 'model'

    model = LudwigModel.load(model_dir, backend='local')
    predictions, output_directory = model.predict(
        dataset=str(test_file),
        skip_save_unprocessed_output=True,
        skip_save_predictions=False,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
