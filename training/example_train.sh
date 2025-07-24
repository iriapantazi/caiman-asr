#! /bin/bash

set -Eeuo pipefail
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
SPM_SIZE=1023
MODEL=testing
CONFIG_NAME="${MODEL}-${SPM_SIZE}sp"
DATASET_NAME_LOWER_CASE=open1.8-beta
WINSZ=0.02 # 0.025 for base/large, 0.02 for testing
MAX_DURATION_SECS=20.0

# TRANSCRIPTS='"meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct-stuctured-1", "transcript"'
TRANSCRIPTS='"transcript"'

cat /workspace/training/configs/${CONFIG_NAME}.yaml |
	sed "s|TRANSCRIPTS|${TRANSCRIPTS}|" |
	sed "s|SENTENCEPIECE|${DATASET_NAME_LOWER_CASE}${SPM_SIZE}|" |
	sed "s|STATS_SUBDIR|${DATASET_NAME_LOWER_CASE}-winsz${WINSZ}|" |
	sed "s/NGRAM_SUBDIR/${DATASET_NAME_LOWER_CASE}${SPM_SIZE}/" |
	sed "s/MAX_DURATION/${MAX_DURATION_SECS}/" |
	sed "s/WORDS/words/" > \
		/tmp/${CONFIG_NAME}.yaml

# Set as appropriate for machine
GBS=1024
GAB=64
BSF=1
NUM_GPUS=$(nvidia-smi -L | wc -l)

./scripts/train.sh \
	--train_manifests librispeech-train-clean-100-flac.eos.json \
	--model_config /tmp/${CONFIG_NAME}.yaml \
	--num_gpus "$NUM_GPUS" \
	--global_batch_size $GBS \
	--grad_accumulation_batches $GAB \
	--batch_split_factor $BSF \
	--val_manifests librispeech-dev-other-flac.json \
	--data_dir /datasets/OpenSourceEOS/ \
	--skip_state_dict_check \
	--training_steps 20 \
	--calculate_emission_latency \
	--delay_penalty 0.0
