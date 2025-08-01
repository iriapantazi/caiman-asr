#! /bin/bash

set -Eeuo pipefail
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
NAME=$(basename "$0".sh)
SPM_SIZE=17407
MODEL=large
CONFIG_NAME="${MODEL}-${SPM_SIZE}sp"
DATASET_NAME_LOWER_CASE=winter2024_fallback
WINSZ=0.025
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
# It is using ~83/97 GB of GPU memory
# GPU utilization reaches 90-100% during training
GBS=1024
GAB=4
BSF=8
NUM_GPUS=1 # $(nvidia-smi -L | wc -l)
# 8 4 -> OK now trying 4 8 (higher effective batch size)
# usign ~70/96 GB of GPU memory

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
	--training_steps 2000 \
	--calculate_emission_latency \
	--val_batch_size 4 \
	--output_dir /results/$NAME \
	--delay_penalty 0.0
