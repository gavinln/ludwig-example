import logging
import pathlib

from ludwig.utils.data_utils import load_json
from ludwig.visualize import learning_curves


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    training_statistics = load_json(
        SCRIPT_DIR
        / 'results'
        / 'api_experiment_run'
        / 'training_statistics.json'
    )
    output_directory = SCRIPT_DIR / 'visualizations'
    list_of_stats = [training_statistics]
    list_of_models = ['api_experiment_run']
    learning_curves(
        list_of_stats,
        output_feature_name='recommended',
        model_names=list_of_models,
        output_directory=output_directory,
        file_format='png'
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
