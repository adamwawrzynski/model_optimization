import torch
import numpy as np
import torch_tensorrt # to install follow https://github.com/pytorch/TensorRT/issues/1371#issuecomment-1256035010 in version 1.3.0
from typing import Tuple
import time
from memory import print_memory_info
import torchmetrics
import torch.nn.utils.prune as prune
from model import CustomLSTM
from torch.jit import ScriptModule
from model_utils import load_model, get_model_name, load_torchscript_model
from dataset_utils import load_dataset


torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level(torch_tensorrt.logging.Level.Error))


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
        mode_label: str = "FP32"
    else:
        model = load_torchscript_model(
            model_torchscript_path=model_torchscript_path,
            device=device,
        )
        mode_label: str = "FP32 JIT"

    inference_time = measure_inference_latency(
        model=model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )

    print(f"[{model_class_name}] {mode_label} Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")


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
        model = load_model(model_name=model_name, device=device, batch_size=batch_size)
        if not use_fp16:
            inference_time = measure_inference_latency(
                model=model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            mode_label: str = "FP32"
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_time = measure_inference_latency(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    dtype="fp16",
                    n_runs=n_runs,
                )
            mode_label: str = "FP16"
    else:
        model = load_torchscript_model(
            model_torchscript_path=model_torchscript_path,
            device=device,
        )
        if not use_fp16:
            inference_time = measure_inference_latency(
                model=model,
                device=device,
                batch_size=batch_size,
                n_runs=n_runs,
            )
            mode_label: str = "FP32 JIT"
        else:
            mode_label: str = "FP16 JIT"
            torch._C._jit_set_autocast_mode(True)
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inference_time = measure_inference_latency(
                        model=model,
                        device=device,
                        batch_size=batch_size,
                        n_runs=n_runs,
                    )
            except RuntimeError:
                print(f"[{model_class_name}] {mode_label} {device.type} Inference: RuntimeError")
                return

    print(f"[{model_class_name}] {mode_label} {device.type} Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
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

    if not use_fp16:
        enabled_precisions = set([torch_tensorrt._enums.dtype.float])
        mode_label: str = "FP32"
    else:
        enabled_precisions = {torch.float16} # run FP16: https://github.com/pytorch/TensorRT/issues/603
        mode_label: str = "FP16"

    if not use_jit:
        model = load_model(model_name=model_name, device=device, batch_size=batch_size).to(device)
    else:
        model = load_torchscript_model(
            model_torchscript_path=model_torchscript_path,
            device=device,
        )

        if not use_fp16:
            mode_label: str = "FP32 JIT"
        else:
            mode_label: str = "FP16 JIT"

    # TensorRT usage based on https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(input_size)],
        enabled_precisions=enabled_precisions,
        workspace_size=1 << 20, # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
        device={
            "device_type": torch_tensorrt.DeviceType.GPU,
            "gpu_id": 0,
        },
    )

    inference_time = measure_inference_latency(
        model=trt_model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] {mode_label} {device.type} TRT Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
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

    inference_time = measure_inference_latency(
        model=trt_pqt_model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] UINT8 {device.type} TRT PQT Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
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

    inference_time = measure_inference_latency(
        model=quantized_model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] UINT8 {device.type} Dynamic Quantization Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
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

    inference_time = measure_inference_latency(
        model=model,
        device=device,
        batch_size=batch_size,
        n_runs=n_runs,
    )
    print(f"[{model_class_name}] FP32 {device.type} Unstructured Pruning (amount={amount}) Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")
    print_memory_info("mb")
