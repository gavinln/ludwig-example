SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


.DEFAULT_GOAL=help
.PHONY: help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: tmux
tmux:  ## start tmux
	tmuxp load tmux.yaml

.PHONY: mypy
mypy:  ## Use mypy type checker
	python -m mypy --ignore-missing-imports --follow-imports=skip \
		--check-untyped-defs .

.PHONY: black
black:  ## apply Python black formatter
	python -m black --line-length 79 .

TEXT_CLF_DIR := $(SCRIPT_DIR)/python/01_text_classification

text_classification:  ## text_classification prepare
	curl -o $(TEXT_CLF_DIR)/reuters-allcats-6.zip http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/reuters-allcats-6.zip
	unzip $(TEXT_CLF_DIR)/reuters-allcats-6.zip -d $(TEXT_CLF_DIR)

text_classification_exp:  ## text classification experiement
	ludwig experiment --data_csv $(TEXT_CLF_DIR)/reuters-allcats.csv \
		--model_definition_file $(TEXT_CLF_DIR)/model_definition_file.yaml

text_classification_plot:  ## text classification plot
	ludwig visualize --visualization learning_curves \
		--training_statistics results/experiment_run/training_statistics.json \
		--file_format png --output_directory results

.PHONY: ray-start-head
ray-start-head:  ## start ray head node
	poetry run ray start --head --port 6380 --num-cpus 2

.PHONY: ray-start-node
ray-start-node:  ## start ray non-head node
	poetry run ray start --address 127.0.0.1:6380 --num-cpus 12

.PHONY: 01-train
01-train:  ## train model 01-getting-started
	python examples/01-getting-started/model-ops.py train-rt

.PHONY: 01-train-zscore
01-train-zscore:  ## train model 01-getting-started with zscore
	python examples/01-getting-started/model-ops.py train-rt-zscore

.PHONY: 01-predict
01-predict:  ## predict 01-getting-started
	python examples/01-getting-started/model-ops.py predict

.PHONY: 01-viz-training
01-viz-training:  ## visualize training for 01-getting-started
	python examples/01-getting-started/model-ops.py visualize-training

.PHONY: 01-compare-perf
01-compare-perf:  ## compare performance for 01-getting-started
	python examples/01-getting-started/model-ops.py compare-perf

.PHONY: 02-train
02-train:  ## train model 02-higgs-tabnet
	python examples/02-higgs-tabnet/model-ops.py train-higgs-small

.PHONY: clean
clean:  # remove temporary files
	rm -rf examples/01-getting-started/output
