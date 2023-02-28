import torch
import numpy as np
import torch_tensorrt # to install follow https://github.com/pytorch/TensorRT/issues/1371#issuecomment-1256035010 in version 1.3.0
from transformers import BatchEncoding
from typing import Union
import time
from abc import abstractmethod, ABC
from dataset_utils import DatasetFactory
from memory import print_memory_info
import transformers
import torchmetrics
import torch.nn.utils.prune as prune
from model import CustomLSTM, T5, GPTNeo
from torch.jit import ScriptModule
from model_utils import load_model, get_model_name, load_torchscript_model
from dataset_utils import CustomDataset


torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level(torch_tensorrt.logging.Level.Error))


def prepare_dataset(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    drop_last: bool,
    device: torch.device,
    dtype: str = "fp32",
):
    X = []
    Y = []
    if not isinstance(dataset, CustomDataset):
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=drop_last,
        )
        dataset = testloader

    for x, y in dataset:
        X.append(x)
        Y.append(y)

    for index, _ in enumerate(X):
        if isinstance(X[index], torch.Tensor):
            X[index]= X[index].to(device, memory_format=torch.channels_last)
        else: # for BatchEncoding
            X[index]= X[index].to(device)

        if dtype == "fp16":
            if isinstance(X[index], torch.Tensor):
                X[index] = X[index].half()

    return X, Y


def get_model_label(
    use_fp16: bool,
    use_jit: bool,
) -> str:
    model_label: str = "FP16" if use_fp16 else "FP32"
    if use_jit:
        model_label = f"{model_label} JIT"

    return model_label


def load_model_based_on_mode(
    model_name: str,
    device: torch.device,
    batch_size: int,
    model_torchscript_path: str,
    use_jit: bool,
):
    if not use_jit:
        model = load_model(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        ).to(device)
    else:
        model = load_torchscript_model(
            model_torchscript_path=model_torchscript_path,
            device=device,
        )
        
    return model


def create_tensorrt_inputs(
    batch_size: int,
    sample: Union[transformers.BatchEncoding, torch.Tensor],
) -> torch_tensorrt.Input:
    if isinstance(sample, BatchEncoding):
        inputs = [torch_tensorrt.Input((batch_size, *(val.shape))) for val in sample.values()]
    elif isinstance(sample, torch.Tensor):
        inputs = [torch_tensorrt.Input((batch_size, *list(sample.shape)))]

    return inputs


def measure_inference_latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    n_runs: int,
    dtype: str = "fp32",
    num_warmups: int = 50,
) -> float:
    # based on https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
    model.to(device, memory_format=torch.channels_last) # improve performance: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api
    model.eval()

    drop_last: bool = False
    # for LSTM model in TorchScript size of network is known in advance
    # changing batch_size will cause dimension mismatch
    if isinstance(model, CustomLSTM) or (isinstance(model, ScriptModule) and model.original_name == "CustomLSTM"):
        drop_last = True

    X, Y = prepare_dataset(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        device=device,
        dtype=dtype,
    )
    num_samples = len(X)

    with torch.no_grad():
        for index in range(num_warmups):
            if isinstance(X[index], BatchEncoding):
                _ = model(**X[index])
            else:
                _ = model(X[index])
    if "cuda" in device.type:
        torch.cuda.synchronize()

    elapsed_time_ave_list = []
    for _ in range(0, n_runs):
        with torch.no_grad():
            Y_pred = []
            start_time = time.time()
            for index, x in enumerate(X):
                if isinstance(x, BatchEncoding):
                    y_pred = model(**x)
                else:
                    y_pred = model(x)
                Y_pred.append(y_pred)
                if "cuda" in device.type:
                    torch.cuda.synchronize()

            end_time = time.time()
            if (not isinstance(model, T5) or (isinstance(model, ScriptModule) and model.original_name == "T5")) and (not isinstance(model, GPTNeo) or (isinstance(model, ScriptModule) and model.original_name == "GPTNeo")):
                predicted_class = []
                for y_pred in Y_pred:
                    predicted_class.append(torch.argmax(y_pred, dim=1))

        if (not isinstance(model, T5) or (isinstance(model, ScriptModule) and model.original_name == "T5")) and (not isinstance(model, GPTNeo) or (isinstance(model, ScriptModule) and model.original_name == "GPTNeo")):
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


class Benchmark(ABC):
    @classmethod
    def measure_vram(cls, unit: str = "mb"):
        print_memory_info(unit=unit)

    @classmethod
    def measure_inference_time(
        cls,
        model_class_name: str,
        mode_label: str,
        inference_time: float,
        batch_size: int,
        n_runs: int,
    ):
        print(f"[{model_class_name}] {mode_label} Inference Latency: {round(inference_time * 1000, 3)} ms / batch ({batch_size} samples) [n_runs={n_runs}]")

    @abstractmethod
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        ...

    def benchmark(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> None:
        inference_time = self.measure_time(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            dataset_factory=dataset_factory,
            model_torchscript_path=model_torchscript_path,
            use_jit=use_jit,
            use_fp16=use_fp16,
            n_runs=n_runs,
            **kwargs,
        )
        model_class_name = get_model_name(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        mode_label = get_model_label(
            use_fp16=use_fp16,
            use_jit=use_jit,
        )
        Benchmark.measure_inference_time(
            model_class_name=model_class_name,
            mode_label=mode_label,
            inference_time=inference_time,
            batch_size=batch_size,
            n_runs=n_runs,
        )
        self.measure_vram()


class BenchmarkCPU(Benchmark):
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        dataset = dataset_factory.get_dataset()
        model = load_model_based_on_mode(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=use_jit,
        )

        inference_time = measure_inference_latency(
            model=model,
            device=device,
            batch_size=batch_size,
            dataset=dataset,
            n_runs=n_runs,
        )
        return inference_time


class BenchmarkCUDA(Benchmark):
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        if use_fp16 and use_jit:
            torch._C._jit_set_autocast_mode(True)

        model = load_model_based_on_mode(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=use_jit,
        )
        dataset = dataset_factory.get_dataset()

        if not use_fp16:
            inference_time = measure_inference_latency(
                model=model,
                device=device,
                batch_size=batch_size,
                dataset=dataset,
                n_runs=n_runs,
            )
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_time = measure_inference_latency(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    dataset=dataset,
                    n_runs=n_runs,
                )

        return inference_time


class BenchmarkTensorRT(Benchmark):
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        dataset = dataset_factory.get_dataset()
        sample =  dataset[0][0]

        inputs = create_tensorrt_inputs(batch_size=batch_size, sample=sample)

        if not use_fp16:
            enabled_precisions = set([torch_tensorrt._enums.dtype.float])
        else:
            enabled_precisions = {torch.float16} # run FP16: https://github.com/pytorch/TensorRT/issues/603

        model = load_model_based_on_mode(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=use_jit,
        )

        # TensorRT usage based on https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
        trt_model = torch_tensorrt.compile(
            module=model,
            inputs=inputs,
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
            dataset=dataset,
            n_runs=n_runs,
        )
        return inference_time


class BenchmarkTensorPTQ(Benchmark):
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> None:
        dataset = dataset_factory.get_dataset()
        sample =  dataset[0][0]

        # PTQ usage based on https://pytorch.org/TensorRT/tutorials/ptq.html#ptq
        model_class_name = get_model_name(model_name=model_name, device=device, batch_size=batch_size)
        model = load_model_based_on_mode(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=use_jit,
        )

        cache_file = f"./{model_class_name}.calibration.cache"

        inputs = create_tensorrt_inputs(batch_size=batch_size, sample=sample)

        testing_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file=cache_file,
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=device,
        )

        trt_pqt_model = torch_tensorrt.compile(
            module=model,
            inputs=inputs,
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
            dataset=dataset,
            n_runs=n_runs,
        )
        return inference_time


class BenchmarkTensorDynamicQuantization(Benchmark):
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> None:
        model = load_model(model_name=model_name, device=device, batch_size=batch_size)
        dataset = dataset_factory.get_dataset()
        quantized_model = torch.quantization.quantize_dynamic(
            model=model,
            qconfig_spec={torch.nn.Conv2d},
            dtype=torch.qint8,
        )

        inference_time = measure_inference_latency(
            model=quantized_model,
            device=device,
            batch_size=batch_size,
            dataset=dataset,
            n_runs=n_runs,
        )
        return inference_time


class BenchmarkTensorPruning(Benchmark):
    def measure_time(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        name: str,
        amount: float,
        structural_pruning: bool = False,
        **kwargs,
    ) -> None:
        model = load_model(model_name=model_name, device=device, batch_size=batch_size)
        dataset = dataset_factory.get_dataset()
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
            dataset=dataset,
            n_runs=n_runs,
        )
        return inference_time
