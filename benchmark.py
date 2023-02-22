import os
import torch
import numpy as np
import torch_tensorrt # to install follow https://github.com/pytorch/TensorRT/issues/1371#issuecomment-1256035010 in version 1.3.0
from torchvision.models import resnet18, ResNet18_Weights, MobileNet_V3_Large_Weights, mobilenet_v3_large, swin_t, Swin_T_Weights, vit_b_16, ViT_B_16_Weights
from typing import Tuple
import time
from memory import print_memory_info
import torchvision
import torchmetrics
from torchvision import transforms
import torch.nn.utils.prune as prune
from model import CustomFCN, CustomCNN, CustomLSTM
from torch.jit import ScriptModule

import argparse


torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level(torch_tensorrt.logging.Level.Error))


def save_torchscript_model(model: torch.nn.Module, model_torchscript_path: str,) -> None:
    model_dir = os.path.dirname(model_torchscript_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.jit.save(torch.jit.script(model), model_torchscript_path)


def load_torchscript_model(model_torchscript_path: str, device: torch.device) -> torch.ScriptModule:
    model = torch.jit.load(model_torchscript_path, map_location=device)
    model = torch.jit.optimize_for_inference(model.eval()) # this line is essential: https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch.jit.optimize_for_inference
    return model


def load_model(model_name: str, device: torch.device, batch_size: int) -> torch.nn.Module:
    if model_name == "swin_t":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    elif model_name == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif model_name == "fcn":
        model = CustomFCN(input_size=3*224*224, hidden_size=224, num_classes=1000)
    elif model_name == "cnn":
        model = CustomCNN(num_classes=1000)
    elif model_name == "rnn":
        model = CustomLSTM(
            input_size=224,
            hidden_size=100,
            layer_size=100,
            num_classes=1000,
            batch_size=batch_size,
            device=device,
        )

    model.eval()
    return model

def get_model_name(model_name: str, device: torch.device, batch_size: int) -> str:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    return model.__class__.__name__


def load_dataset(use_train: bool = False) -> torchvision.datasets.DatasetFolder:
    # # dataset downloaded from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
    dir_name: str = "val"
    if use_train:
        dir_name = "train"

    data_dir: str = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    testing_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "imagenet-mini", dir_name),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    return testing_dataset


def measure_inference_latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    n_runs: int,
    dtype: str = "fp32",
    num_warmups: int = 50,
) -> float:
    # based on https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
    model.to(device, memory_format=torch.channels_last) # improve performance: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api
    model.eval()

    testing_dataset = load_dataset()

    drop_last: bool = False
    # for LSTM model in TorchScript size of network is known in advance
    # changing batch_size will cause dimension mismatch
    if isinstance(model, CustomLSTM) or (isinstance(model, ScriptModule) and model.original_name == "CustomLSTM"):
        drop_last = True

    testloader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=drop_last,
    )
    X = []
    Y = []
    for x, y in testloader:
        X.append(x)
        Y.append(y)

    num_samples = len(X)
    for index, _ in enumerate(X):
        X[index]= X[index].to(device, memory_format=torch.channels_last)
        if dtype == "fp16":
            X[index] = X[index].half()

    with torch.no_grad():
        for index in range(num_warmups):
            _ = model(X[index])
    if "cuda" in device.type:
        torch.cuda.synchronize()

    elapsed_time_ave_list = []
    for _ in range(0, n_runs):
        with torch.no_grad():
            Y_pred = []
            start_time = time.time()
            for index, x in enumerate(X):
                y_pred = model(x)
                Y_pred.append(y_pred)
                if "cuda" in device.type:
                    torch.cuda.synchronize()

            end_time = time.time()
            predicted_class = []
            for y_pred in Y_pred:
                predicted_class.append(torch.argmax(y_pred, dim=1))

        f1 = torchmetrics.F1Score(task="multiclass", num_classes=1000)
        preds = torch.hstack(predicted_class).flatten().cpu().detach()
        labels = torch.hstack(Y).flatten()
        min_length = min(preds.shape[0], labels.shape[0])
        score = f1(preds[:min_length], labels[:min_length])
        score_rounded = round(score.cpu().detach().item(), 3)
        print(f"F1 score: {score_rounded}")

        elapsed_time = end_time - start_time
        elapsed_time_ave = elapsed_time / (num_samples * batch_size)

        elapsed_time_ave_list.append(elapsed_time_ave)


    return np.mean(elapsed_time_ave_list)


def benchmark_cpu(
    model_name: str,
    device: torch.device,
    batch_size: int,
    model_torchscript_path: str,
    use_jit: bool,
    n_runs: int,
) -> None:
    model_class_name = get_model_name(model_name=model_name, device=device, batch_size=batch_size)
    if not use_jit:
        model = load_model(model_name=model_name, device=device, batch_size=batch_size)
        fp32_cpu_inference_latency = measure_inference_latency(
            model=model,
            device=device,
            batch_size=batch_size,
            n_runs=n_runs,
        )
        print(f"[{model_class_name}] FP32 {device.type} Inference Latency: {round(fp32_cpu_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
    else:
        model_cpu_jit = load_torchscript_model(
            model_torchscript_path=model_torchscript_path,
            device=device,
        )
        fp32_cpu_inference_latency = measure_inference_latency(
            model=model_cpu_jit,
            device=device,
            batch_size=batch_size,
            n_runs=n_runs,
        )
        print(f"[{model_class_name}] FP32 {device.type} JIT Inference Latency: {round(fp32_cpu_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")


def benchmark_cuda(
    model_name: str,
    device: torch.device,
    batch_size: int,
    model_torchscript_path: str,
    use_jit: bool,
    use_fp16: bool,
    n_runs: int,
) -> None:
    model_class_name = get_model_name(model_name=model_name, device=device, batch_size=batch_size)
    if not use_jit:
        if not use_fp16:
            model = load_model(model_name=model_name, device=device, batch_size=batch_size)
            fp32_cuda_inference_latency = measure_inference_latency(
                model=model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            print(f"[{model_class_name}] FP32 {device.type} Inference Latency: {round(fp32_cuda_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")
        else:
            model = load_model(model_name=model_name, device=device, batch_size=batch_size)
            model_class_name = model.__class__.__name__
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                fp16_cuda_inference_latency = measure_inference_latency(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    dtype="fp16",
                    n_runs=n_runs,
                )
            print(f"[{model_class_name}] FP16 {device.type} Inference Latency: {round(fp16_cuda_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")

    else:
        if not use_fp16:
            model_cuda_jit = load_torchscript_model(
                model_torchscript_path=model_torchscript_path,
                device=device,
            )
            fp32_cuda_inference_latency = measure_inference_latency(
                model=model_cuda_jit,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            print(f"[{model_class_name}] FP32 {device.type} JIT Inference Latency: {round(fp32_cuda_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")
        else:
            torch._C._jit_set_autocast_mode(True)
            model_cuda_jit = load_torchscript_model(
                model_torchscript_path=model_torchscript_path,
                device=device,
            )
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fp16_cuda_inference_latency = measure_inference_latency(
                        model=model_cuda_jit,
                        device=device,
                        batch_size=batch_size,
                        n_runs=n_runs,
                    )
                print(f"[{model_class_name}] FP16 {device.type} JIT Inference Latency: {round(fp16_cuda_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            except RuntimeError:
                print(f"[{model_class_name}] FP16 {device.type} JIT Inference: RuntimeError")

            print_memory_info("mb")


def benchmark_tensorrt(
    model_name: str,
    device: torch.device,
    batch_size: int,
    input_size: Tuple[int, int, int, int],
    model_torchscript_path: str,
    use_jit: bool,
    use_fp16: bool,
    n_runs: int,
) -> None:
    model_class_name = get_model_name(model_name=model_name, device=device, batch_size=batch_size)
    if not use_jit:
        if not use_fp16:
            model = load_model(model_name=model_name, device=device, batch_size=batch_size).to(device)
            # TensorRT usage based on https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(input_size)],
                workspace_size=1 << 20, # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
                device={
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0,
                },
            )
            fp32_trt_inference_latency = measure_inference_latency(
                model=trt_model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            print(f"[{model_class_name}] FP32 {device.type} TRT Inference Latency: {round(fp32_trt_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")
        else:
            model = load_model(model_name=model_name, device=device, batch_size=batch_size).to(device)
            trt_fp16_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(input_size)],
                enabled_precisions={torch.float16},  # run FP16: https://github.com/pytorch/TensorRT/issues/603
                workspace_size=1 << 20, # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
                device={
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0,
                },
            )
            fp16_cuda_inference_latency = measure_inference_latency(
                model=trt_fp16_model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            print(f"[{model_class_name}] FP16 {device.type} TRT Inference Latency: {round(fp16_cuda_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")

    else:
        if not use_fp16:
            model_cuda_jit = load_torchscript_model(
                model_torchscript_path=model_torchscript_path,
                device=device,
            )
            trt_jit_model = torch_tensorrt.compile(
                model_cuda_jit,
                inputs=[torch_tensorrt.Input(input_size)],
                workspace_size=1 << 20, # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
                device={
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0,
                },
            )
            fp32_jit_trt_inference_latency = measure_inference_latency(
                model=trt_jit_model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            print(f"[{model_class_name}] FP32 {device.type} TRT JIT Inference Latency: {round(fp32_jit_trt_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")
        else:
            model_cuda_jit = load_torchscript_model(
                model_torchscript_path=model_torchscript_path,
                device=device,
            )
            trt_jit_fp16_model = torch_tensorrt.compile(
                model_cuda_jit,
                inputs=[torch_tensorrt.Input(input_size)],
                enabled_precisions={torch.float16},  # run FP16: https://github.com/pytorch/TensorRT/issues/603
                workspace_size=1 << 20, # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
                device={
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0,
                },
            )
            fp16_cuda_jit_inference_latency = measure_inference_latency(
                model=trt_jit_fp16_model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            print(f"[{model_class_name}] FP16 {device.type} TRT JIT Inference Latency: {round(fp16_cuda_jit_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
            print_memory_info("mb")


def benchmark_tensorrt_ptq(
    model_name: str,
    device: torch.device,
    batch_size: int,
    input_size: Tuple[int, int, int, int],
    n_runs: int,
) -> None:
    # PTQ usage based on https://pytorch.org/TensorRT/tutorials/ptq.html#ptq
    model = load_model(model_name=model_name, device=device, batch_size=batch_size).to(device)
    model_class_name = model.__class__.__name__
    cache_file = f"./{model_class_name}.calibration.cache"

    testing_dataset = load_dataset(use_train=False)
    testing_dataloader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        testing_dataloader,
        cache_file=cache_file,
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=device,
    )

    trt_pqt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(input_size)],
        enabled_precisions={torch.int8},
        calibrator=calibrator,
        workspace_size=1 << 20, # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
        device={
            "device_type": torch_tensorrt.DeviceType.GPU,
            "gpu_id": 0,
            "dla_core": 0,
            "allow_gpu_fallback": False,
            "disable_tf32": False
        })
    del calibrator

    fp32_trt_inference_latency = measure_inference_latency(
        model=trt_pqt_model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] UINT8 {device.type} TRT PQT Inference Latency: {round(fp32_trt_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
    print_memory_info("mb")


def benchmark_dynamic_quantization(
    model_name: str,
    device: torch.device,
    batch_size: int,
    n_runs: int,
) -> None:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    model_class_name = model.__class__.__name__
    quantized_model = torch.quantization.quantize_dynamic(
        model=model,
        qconfig_spec={torch.nn.Conv2d},
        dtype=torch.qint8,
    )

    fp32_dq_inference_latency = measure_inference_latency(
        model=quantized_model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] UINT8 {device.type} Dynamic Quantization Inference Latency: {round(fp32_dq_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
    print_memory_info("mb")


def benchmark_pruning(
    model_name: str,
    device: torch.device,
    batch_size: int,
    n_runs: int,
    name: str,
    amount: float,
    structural_pruning: bool = False,
) -> None:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    model_class_name = model.__class__.__name__
    module_set = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            module_set.add((module, "weight"))
            if structural_pruning:
                prune.ln_structured(
                    module=module,
                    name=name,
                    amount=amount,
                    n=2,
                    dim=0,
                )

    if not structural_pruning:
        prune.global_unstructured(
            parameters=module_set,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    fp32_dq_inference_latency = measure_inference_latency(
        model=model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] FP32 {device.type} Unstructured Pruning (amount={amount}) Inference Latency: {round(fp32_dq_inference_latency * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
    print_memory_info("mb")


def save_torchscript(
    model_name: str,
    device: torch.device,
    batch_size: int,
    model_torchscript_path: str,
) -> None:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    save_torchscript_model(model=model, model_torchscript_path=model_torchscript_path)
    del model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark model optimization techniques")
    parser.add_argument("--type", choices=["cpu", "cuda", "tensorrt", "quantization", "dynamic_quantization", "pruning"], required=True, help="Model's operation type.")
    parser.add_argument("--model_name", choices=["swin_t", "vit", "resnet", "mobilenet", "fcn", "cnn", "rnn"], required=True, help="Model's name.")
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

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    input_size = (args.batch_size, 3, 224, 224) # dimensions for ImageNet samples

    # save model's torchscript .pth file
    model_torchscript_path: str = os.path.join(args.model_dir, args.model_filename)
    save_torchscript(
        model_name=args.model_name,
        device=cpu_device,
        batch_size=args.batch_size,
        model_torchscript_path=model_torchscript_path,
    )

    # compute inference time, CUDA memory usage and F1 score
    if args.type == "cpu":
        benchmark_cpu(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            n_runs=args.n_runs,
        )
    elif args.type == "cuda":
        benchmark_cuda(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
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
            input_size=input_size,
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
            input_size=input_size,
            n_runs=args.n_runs,
        )
    elif args.type == "dynamic_quantization":
        benchmark_dynamic_quantization(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            n_runs=args.n_runs,
        )
    elif args.type == "pruning":
        benchmark_pruning(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            name="weight",
            amount=args.pruning_ratio,
            n_runs=args.n_runs,
            structural_pruning=args.structural_pruning,
        )


if __name__ == "__main__":
    main()
