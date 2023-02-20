#!/bin/bash

MODEL_NAME="resnet"
N_RUNS="5"

for BATCH_SIZE in "1" "16" "32" "64"; do
    poetry run python3 benchmark.py --type cpu --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type cpu --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type cuda --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type cuda --use_fp$BATCH_SIZE6 --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type cuda --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type cuda --use_fp$BATCH_SIZE6 --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type tensorrt --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type tensorrt --use_fp$BATCH_SIZE6 --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type tensorrt --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type tensorrt --use_fp$BATCH_SIZE6 --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type quantization --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 benchmark.py --type dynamic_quantization --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
done
