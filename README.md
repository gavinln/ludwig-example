# Ludwig machine learning library from Uber

Examples of using the [Ludwig][10] library from Uber.

[10]: https://uber.github.io/ludwig/

## Install software

```
nix-shell
```

### Install Ludwig

1. Install Pip

```
# python 3.6.9
sudo apt install python3-pip
python3 -m venv ~/.ludwig-example
source ~/.ludwig-example/bin/activate
which python
pip install wheel
pip install ludwig[full]
```

### Install other software

1. Install pre-requisites: [gmpy][200] - GNU multi-precision libraries

[200]: https://stackoverflow.com/questions/40075271/gmpy2-not-installing-mpir-h-not-found

```
sudo apt-get install -y libgmp-dev
sudo apt-get install -y libmpfr-dev
sudo apt-get install -y libmpc-dev
```

2. Install libraries without locking dependencies

```
pipenv install ludwig --skip-lock
```

3. Lock the dependencies

```
export PIP_DEFAULT_TIMEOUT=600
pipenv lock
```

## Examples



## Links

### Ludwig documentation

[Ludwig github page][1000]

[1000]: https://github.com/uber/ludwig

[Ludwig documentation][1010]

[1010]: https://uber.github.io/ludwig/

[Ludwig 0.2 version release][1020]

[1020]: https://eng.uber.com/ludwig-v0-2/

### Ludwig tutorials

[AutoML with Ludwig][1100]

[1100]: http://datasmarts.net/automl-with-ludwig/

[Guide to Ludwig][1110]

[1110]: https://blog.dominodatalab.com/a-practitioners-guide-to-deep-learning-with-ludwig/

### Tensorflow links

[Tensorflow][1200] library locations list the different versions available.

[1200]: https://www.tensorflow.org/install/pip#package-location

## Miscellaneous

### Send to IPython terminal using neoterm

1. Type %autoindent in the IPython terminal and make sure Automatic indentation
   is toggled OFF

