# Ludwig machine learning library from Uber

Examples of using the [Ludwig][100] library from Uber.

[100]: https://github.com/ludwig-ai/ludwig

## Setup code

1. Clone the repository
git clone https://github.com/gavinln/ludwig-example

## Examples

### 01-getting-started

This examples using the Rotten Tomatoes dataset which is a CSV file with a
variety of feature types and a binary target.

#### Setup data

1. Change to the example root directory

```
pushd examples/01-getting-started/
```

2. Download the training data

```
curl -OL https://ludwig-ai.github.io/ludwig-docs/0.5/data/rotten_tomatoes.csv
```

3. Download the test data

```
curl -OL https://ludwig-ai.github.io/ludwig-docs/0.5/data/rotten_tomatoes_test.csv
```

3. Change to the project root

```
popd
```

#### Train model

make 01-getting-started

#### Use model to predit

https://ludwig-ai.github.io/ludwig-docs/0.5/getting_started/prepare_data/

## Links

### Ludwig documentation

[Ludwig documentation][900]

[900]: https://ludwig-ai.github.io/ludwig-docs/0.5/

[Ludwig on PyTorch][910]

[910]: https://medium.com/pytorch/ludwig-on-pytorch-1241776417fc

#### Videos

https://www.youtube.com/watch?v=5unXta3CYSw

https://www.youtube.com/watch?v=nV3uWgmGjvY

https://www.youtube.com/watch?v=74hqlj5k4Zg

https://www.youtube.com/watch?v=BGRwMq0Wc5M
