#! /bin/bash

set -Eeuo pipefail
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
SPM_SIZE=1023
MODEL=testing
CONFIG_NAME="${MODEL}-${SPM_SIZE}sp"
DATASET_NAME_LOWER_CASE=open1.8-beta
WINSZ=0.02
MAX_DURATION_SECS=20.0

TRANSCRIPTS='"transcript"'

cat configs/${CONFIG_NAME}.yaml |
	sed "s|TRANSCRIPTS|${TRANSCRIPTS}|" |
	sed "s|SENTENCEPIECE|${DATASET_NAME_LOWER_CASE}${SPM_SIZE}|" |
	sed "s|STATS_SUBDIR|${DATASET_NAME_LOWER_CASE}-winsz${WINSZ}|" |
	sed "s/NGRAM_SUBDIR/${DATASET_NAME_LOWER_CASE}${SPM_SIZE}/" |
	sed "s/MAX_DURATION/${MAX_DURATION_SECS}/" > \
		/tmp/${CONFIG_NAME}.yaml

# Set as appropriate for machine
GBS=1024
GAB=64
BSF=1
NUM_GPUS=$(nvidia-smi -L | wc -l)

./scripts/train.sh \
	--train_manifests librispeech-dev-clean-flac.json \
	--model_config /tmp/${CONFIG_NAME}.yaml \
	--num_gpus "$NUM_GPUS" \
	--global_batch_size $GBS \
	--grad_accumulation_batches $GAB \
	--batch_split_factor $BSF \
	--val_manifests librispeech-dev-clean-flac.json \
	--data_dir /datasets/LibriSpeech/ \
	--skip_state_dict_check \
	--training_steps 20 \
	--calculate_emission_latency \
	--delay_penalty 0.0
