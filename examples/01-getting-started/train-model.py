import logging
import pathlib

from ludwig.api import LudwigModel
import pandas


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
log = logging.getLogger(__name__)


def main():
    df = pandas.read_csv(SCRIPT_DIR / 'rotten_tomatoes.csv')
    print(f'data shape: {df.shape=}')
    model = LudwigModel(config=str(SCRIPT_DIR / 'rotten_tomatoes.yaml'))
    results = model.train(dataset=df)
    print(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
