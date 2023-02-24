import os
import torch
import torch_tensorrt # to install follow https://github.com/pytorch/TensorRT/issues/1371#issuecomment-1256035010 in version 1.3.0
from benchmark import benchmark_cpu, benchmark_cuda, benchmark_tensorrt, benchmark_tensorrt_ptq, benchmark_dynamic_quantization, benchmark_pruning
from model_utils import save_torchscript
from dataset_utils import load_dataset_cv, load_dataset_nlp
from transformers import BatchEncoding

import argparse

torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level(torch_tensorrt.logging.Level.Error))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark model optimization techniques")
    parser.add_argument("--type", choices=["cpu", "cuda", "tensorrt", "quantization", "dynamic_quantization", "pruning"], required=True, help="Model's operation type.")
    parser.add_argument("--model_name", choices=["swin_t", "vit", "resnet", "mobilenet", "fcn", "cnn", "rnn", "bert", "t5"], required=True, help="Model's name.")
    parser.add_argument("--use_fp16", action="store_true", help="Use half precision model.")
    parser.add_argument("--use_jit", action="store_true", help="Use JIT model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Size of processed batch.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to compute mean of inference times.")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="Directory with saved JIT models.")
    parser.add_argument("--model_filename", type=str, default="model_jit.pth", help="JIT model file name.")
    parser.add_argument("--pruning_ratio", type=float, default=0.2, help="Ratio of model's pruned weights.")
    parser.add_argument("--structural_pruning", action="store_true", help="Use structural pruning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True # has influence on performance on CNNs: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner

    # https://pytorch.org/docs/stable/amp.html

    # For now, we suggest to disable the Jit Autocast Pass,
    # As the issue: https://github.com/pytorch/pytorch/issues/75956
    if args.use_jit:
        torch._C._jit_set_autocast_mode(False)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected. Exiting...")

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    if args.model_name in ["bert", "t5"]:
        load_dataset_func = load_dataset_nlp
    else:
        load_dataset_func = load_dataset_cv

    dataset = load_dataset_func()
    example_inputs = dataset[0][0]
    if isinstance(example_inputs, BatchEncoding):
        example_inputs = [
            example_inputs["input_ids"],
            example_inputs["token_type_ids"],
            example_inputs["attention_mask"],
        ]
    else:
        example_inputs = None

    # save model's torchscript .pth file
    model_torchscript_path: str = os.path.join(args.model_dir, args.model_filename)
    save_torchscript(
        model_name=args.model_name,
        device=cpu_device,
        batch_size=args.batch_size,
        model_torchscript_path=model_torchscript_path,
        example_inputs=example_inputs,
    )

    # compute inference time, CUDA memory usage and F1 score
    if args.type == "cpu":
        benchmark_cpu(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            load_dataset_func=load_dataset_func,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            n_runs=args.n_runs,
        )
    elif args.type == "cuda":
        benchmark_cuda(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
            load_dataset_func=load_dataset_func,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
        )
    elif args.type == "tensorrt":
        benchmark_tensorrt(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
            load_dataset_func=load_dataset_func,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
        )
    elif args.type == "quantization":
        benchmark_tensorrt_ptq(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
            load_dataset_func=load_dataset_func,
            n_runs=args.n_runs,
            use_jit=args.use_jit,
            model_torchscript_path=model_torchscript_path,
        )
    elif args.type == "dynamic_quantization":
        benchmark_dynamic_quantization(
            model_name=args.model_name,
            device=cpu_device,
            load_dataset_func=load_dataset_func,
            batch_size=args.batch_size,
            n_runs=args.n_runs,
        )
    elif args.type == "pruning":
        benchmark_pruning(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            load_dataset_func=load_dataset_func,
            name="weight",
            amount=args.pruning_ratio,
            n_runs=args.n_runs,
            structural_pruning=args.structural_pruning,
        )


if __name__ == "__main__":
    main()
