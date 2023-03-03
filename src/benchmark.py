# pylint: disable = (missing-module-docstring)

import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch

# to install version 1.3.0 follow
# https://github.com/pytorch/TensorRT/issues/1371#issuecomment-1256035010
import torch_tensorrt
import torchmetrics
import transformers
from torch.jit import ScriptModule
from torch.nn.utils import prune
from transformers import BatchEncoding

from src.dataset_utils import CustomDataset, DatasetFactory
from src.memory import print_memory_info
from src.model import T5, CustomLSTM, GPTNeo
from src.model_utils import get_model_name, load_model, load_torchscript_model

torch_tensorrt.logging.set_reportable_log_level(
    torch_tensorrt.logging.Level(torch_tensorrt.logging.Level.Error)
)


def prepare_dataset(
    dataset: Union[torch.utils.data.Dataset, torch.utils.data.DataLoader],
    batch_size: int,
    drop_last: bool,
    device: torch.device,  # pylint: disable = (no-member)
    dtype: str = "fp32",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    samples: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    data_iterator: Union[torch.utils.data.DataLoader, CustomDataset]
    if isinstance(dataset, CustomDataset):
        data_iterator = dataset
    else:
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=drop_last,
        )
        data_iterator = testloader

    for sample_batch, label_batch in data_iterator:
        samples.append(sample_batch)
        labels.append(label_batch)

    for index, _ in enumerate(samples):
        if isinstance(samples[index], torch.Tensor):
            samples[index] = samples[index].to(device)
            samples[index] = samples[index].to(
                memory_format=torch.channels_last
            )  # pylint: disable = (no-member)
        else:  # for BatchEncoding
            samples[index] = samples[index].to(device)

        if dtype == "fp16":
            if isinstance(samples[index], torch.Tensor):
                samples[index] = samples[index].half()

    return samples, labels


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
    device: torch.device,  # pylint: disable = (no-member)
    batch_size: int,
    model_torchscript_path: str,
    use_jit: bool,
) -> Union[torch.nn.Module, torch._C.ScriptModule]:
    model: Union[torch.nn.Module, torch._C.ScriptModule]
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
        inputs = [
            torch_tensorrt.Input((batch_size, *(val.shape))) for val in sample.values()
        ]
    elif isinstance(sample, torch.Tensor):
        inputs = [torch_tensorrt.Input((batch_size, *list(sample.shape)))]

    return inputs


def measure_inference_latency(
    model: Union[torch.nn.Module, torch._C.ScriptModule],
    device: torch.device,  # pylint: disable = (no-member)
    batch_size: int,
    dataset: torch.utils.data.Dataset,
    n_runs: int,
    dtype: str = "fp32",
    num_warmups: int = 5,
) -> float:
    # https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/

    # improve performance:
    # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#memory-format-api
    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.to(memory_format=torch.channels_last)  # pylint: disable = (no-member)
        model.eval()

    drop_last: bool = False
    # for LSTM model in TorchScript size of network is known in advance
    # changing batch_size will cause dimension mismatch
    if isinstance(model, CustomLSTM) or (
        isinstance(model, ScriptModule) and model.original_name == "CustomLSTM"
    ):
        drop_last = True

    sample_batches, label_batches = prepare_dataset(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        device=device,
        dtype=dtype,
    )
    num_samples = len(sample_batches)
    if num_samples < num_warmups:
        print(
            "WARNING: Number of warmup steps is lower than number of data samples."
            + "Use number of samples instead."
        )
        num_warmups = num_samples

    warmup_model(model, device, num_warmups, sample_batches)

    is_t5_model: bool = isinstance(model, T5) or (
        isinstance(model, ScriptModule) and model.original_name == "T5"
    )
    is_gpt_model: bool = isinstance(model, GPTNeo) or (
        isinstance(model, ScriptModule) and model.original_name == "GPTNeo"
    )
    is_nlg_model: bool = is_t5_model or is_gpt_model

    elapsed_time_ave_list: List[float] = []
    for _ in range(0, n_runs):
        elapsed_time = measure_inference_run(
            model, device, sample_batches, label_batches, is_nlg_model
        )
        elapsed_time_ave_list.append(elapsed_time / (num_samples * batch_size))

    mean_inference_time: float = 0.0
    if elapsed_time_ave_list:
        mean_inference_time = float(np.mean(elapsed_time_ave_list))

    return mean_inference_time


def warmup_model(
    model: Union[torch.nn.Module, torch._C.ScriptModule],
    device: torch.device,  # pylint: disable = (no-member)
    num_warmups: int,
    sample_batches: List[Union[torch.Tensor, BatchEncoding]],
) -> None:
    with torch.no_grad():
        for index in range(num_warmups):
            if isinstance(sample_batches[index], BatchEncoding):
                _ = model(**sample_batches[index])
            else:
                _ = model(sample_batches[index])
    if "cuda" in device.type:
        torch.cuda.synchronize()


def measure_inference_run(
    model: Union[torch.nn.Module, torch._C.ScriptModule],
    device: torch.device,  # pylint: disable = (no-member)
    sample_batches: List[torch.Tensor],
    label_batches: List[torch.Tensor],
    is_nlg_model: bool,
) -> float:
    with torch.no_grad():
        predicted_class: List[torch.Tensor] = []
        start_time = time.time()
        for sample in sample_batches:
            if isinstance(sample, BatchEncoding):
                y_pred = model(**sample)
            else:
                y_pred = model(sample)

            if "cuda" in device.type:
                torch.cuda.synchronize()

            if not is_nlg_model:
                predicted_class.append(torch.argmax(y_pred, dim=1))

    end_time = time.time()

    if not is_nlg_model:
        f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=1000)
        preds = (
            torch.hstack(predicted_class)  # pylint: disable = (no-member)
            .flatten()
            .cpu()
            .detach()
        )
        labels = torch.hstack(label_batches).flatten()  # pylint: disable = (no-member)
        min_length = min(preds.shape[0], labels.shape[0])
        score = f1_metric(  # pylint: disable = (not-callable)
            preds[:min_length], labels[:min_length]
        )
        score_rounded = round(score.cpu().detach().item(), 3)
        print(f"F1 score: {score_rounded}")
    return end_time - start_time


class Benchmark(ABC):
    """Abstract benchmark class."""

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
        inference_time_rounded = round(inference_time * 1000, 3)
        model_label = f"[{model_class_name}] {mode_label}"
        options = f"ms / batch ({batch_size} samples) [n_runs={n_runs}]"
        print(f"{model_label} Inference Latency: {inference_time_rounded} {options}")

    @abstractmethod
    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
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
        device: torch.device,  # pylint: disable = (no-member)
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
    """CPU benchmark class."""

    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
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
    """CUDA benchmark class."""

    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        if use_fp16 and use_jit:
            torch._C._jit_set_autocast_mode(  # pylint: disable = (protected-access,c-extension-no-member)
                True
            )

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
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,  # pylint: disable = (no-member)
            ):
                inference_time = measure_inference_latency(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    dataset=dataset,
                    n_runs=n_runs,
                )

        return inference_time


class BenchmarkTensorRT(Benchmark):
    """TensorRT benchmark class."""

    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        dataset = dataset_factory.get_dataset()
        sample = dataset[0][0]

        inputs = create_tensorrt_inputs(batch_size=batch_size, sample=sample)

        if not use_fp16:
            enabled_precisions = set(
                [
                    torch_tensorrt._enums.dtype.float  # pylint: disable = (protected-access)
                ]
            )
        else:
            # run FP16: https://github.com/pytorch/TensorRT/issues/603
            enabled_precisions = {torch.float16}  # pylint: disable = (no-member)

        model = load_model_based_on_mode(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=use_jit,
        )

        # https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
        trt_model = torch_tensorrt.compile(
            module=model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=1
            << 20,  # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
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
    """TensorRT PTQ benchmark class."""

    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        dataset = dataset_factory.get_dataset()
        sample = dataset[0][0]

        # PTQ usage based on https://pytorch.org/TensorRT/tutorials/ptq.html#ptq
        model_class_name = get_model_name(
            model_name=model_name, device=device, batch_size=batch_size
        )
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
            enabled_precisions={torch.int8},  # pylint: disable = (no-member)
            calibrator=calibrator,
            workspace_size=1
            << 20,  # prevent OutOfMemory error logs: https://github.com/pytorch/TensorRT/issues/603
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        )
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
    """Dynamic Quantization benchmark class."""

    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        model = load_model(model_name=model_name, device=device, batch_size=batch_size)
        dataset = dataset_factory.get_dataset()
        quantized_model = torch.quantization.quantize_dynamic(
            model=model,
            qconfig_spec={torch.nn.Conv2d},
            dtype=torch.qint8,  # pylint: disable = (no-member)
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
    """Pruning benchmark class."""

    def measure_time(
        self,
        model_name: str,
        device: torch.device,  # pylint: disable = (no-member)
        batch_size: int,
        dataset_factory: DatasetFactory,
        model_torchscript_path: str,
        use_jit: bool,
        use_fp16: bool,
        n_runs: int,
        **kwargs,
    ) -> float:
        name: str = kwargs["name"]
        amount: float = kwargs["amount"]
        structural_pruning: bool = kwargs.get("structural_pruning", False)

        model = load_model(model_name=model_name, device=device, batch_size=batch_size)
        dataset = dataset_factory.get_dataset()
        module_set = set()
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
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
