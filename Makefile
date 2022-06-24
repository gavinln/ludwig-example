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


# text_clf_dir
# 01_text_classification/
# named_entity_dir
# 02_named_entity_recognition_tagging/
# 03_natural_language_understanding/
# 04_machine_translation/
# 05_chit_chat_dialogue_modeling_through_sequence2sequence/
# 06_sentiment_analysis/
# 07_image_classification/
# 08_image_classification_mnist/
# 09_download_the_mnist_dataset/
# 10_create_train_and_test_csvs/
# 11_train_a_model/
# 12_image_captioning/
# 13_one-shot_learning_with_siamese_networks/
# 14_visual_question_answering/
# 15_spoken_digit_speech_recognition/
# 16_download_the_free_spoken_digit_dataset/
# 17_create_an_experiment_csv/
# 18_train_a_model/
# 19_speaker_verification/
# 20_kaggles_titanic_predicting_survivors/
# 21_time_series_forecasting/
# 22_time_series_forecasting_weather_data_example/
# 23_movie_rating_prediction/
# 24_multi_label_classification/
# 25_multi_task_learning/
# 26_simple_regression_fuel_efficiency_prediction/
# 27_binary_classification_fraud_transactions_identification/
