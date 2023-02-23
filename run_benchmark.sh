#!/bin/bash

MODEL_NAME="resnet"
N_RUNS="5"

for BATCH_SIZE in "64"; do
    poetry run python3 main.py --type cpu --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type cpu --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type cuda --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type cuda --use_fp16 --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type cuda --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type cuda --use_fp16 --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type tensorrt --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type tensorrt --use_fp16 --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type tensorrt --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type tensorrt --use_fp16 --use_jit --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type quantization --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
    poetry run python3 main.py --type dynamic_quantization --batch_size $BATCH_SIZE --n_runs $N_RUNS --model_name $MODEL_NAME
done
