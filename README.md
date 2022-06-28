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

#### Work with model

1. Train model

```
make 01-train
```

2. Train alternative model

```
make 01-train-zscore
```

3. Predict with the model

```
make 01-predict
```

4. Visualize training of the model

```
make 01-viz-training
```

5. Compare performance of the two models

```
make 01-compare-perf
```

### Troubleshooting

1. Python package tkinter

If you see the error `ModuleNotFoundError: No module named '_tkinter'`

```
brew install python-tk  # Mac
```

https://ludwig-ai.github.io/ludwig-docs/0.5/getting_started/prepare_data/

## Links

### Ludwig documentation

[Ludwig documentation][900]

[900]: https://ludwig-ai.github.io/ludwig-docs/0.5/

[Ludwig on PyTorch][910]

[910]: https://medium.com/pytorch/ludwig-on-pytorch-1241776417fc

### Declarative Machine Learning with Ludwig

#### Videos

https://www.youtube.com/watch?v=BTkl_qc0Plc

https://www.youtube.com/watch?v=5unXta3CYSw

https://www.youtube.com/watch?v=nV3uWgmGjvY

https://www.youtube.com/watch?v=74hqlj5k4Zg

https://www.youtube.com/watch?v=BGRwMq0Wc5M

#### Audio

Interview with the creator of Ludwig

https://www.pythonpodcast.com/ludwig-horovod-distributed-declarative-deep-learning-episode-341/

* Data Scientists new to deep learning but familiar with regression, gradient boosting, etc.
* Software engineers interested in machine learning
* Changing from Tensorflow to PyTorch should not affect users of Ludwig

#### Experts vs Novices

* Novices use Ludwig to train, make predictions and serve models
* Experts use Ludwig as a baseline.

#### Other articles

https://queue.acm.org/detail.cfm?id=3479315

https://medium.com/pytorch/ludwig-on-pytorch-1241776417fc

https://thesequence.substack.com/p/molino

https://blog.dominodatalab.com/a-practitioners-guide-to-deep-learning-with-ludwig

https://medium.com/pytorch/ludwig-on-pytorch-1241776417fc

https://flaven.fr/2021/04/using-ludwig-introduction-to-deep-learning/
