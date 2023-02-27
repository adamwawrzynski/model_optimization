# Model's optimizations

This repository contains scripts to perform a benchmark of different ways to optimize
neural network model's inference. The benchmark reports inference time per single
sample, VRAM requirements and precision of a model.

## Install

To install all dependencies run `poetry install` command.

## Dataset

The benchmark uses the ImageNet-mini dataset, which can be downloaded from the site:
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000. After downloading, extract
the archive in to a directory of a directory named `data/` located in the directory
with the cloned repository.

## Run

To run benchmark run:
```bash
bash run_benchmark.sh
```

If You want to modify a number of iterations or the neural network model modify a variable
at the top of the `run_benchmark.sh` script. The content of this script is as follows:

```bash
#!/bin/bash

MODEL_NAME="resnet"
PRETRAINED_MODEL_NAME="textattack/bert-base-uncased-imdb"
N_RUNS="5"
...
```

## Conclusions

### VRAM memory usage

To minimize the model size on the GPU use only a `fp16` optimization from `amp`. With this
optimization the model is ~2 times smaller.

### Quantization

The TensorRT `INT8` quantization gives the greatest acceleration of the model inference,
but results in a noticeable decrease in the model accuracy. The decrease is greater
the smaller the neural network.


## Results

 Benchmark environment:
* Torch-TensorRT Version (e.g. 1.0.0): 1.3.0
* PyTorch Version (e.g. 1.0): 1.13.1
* CPU Architecture: AMD® Ryzen 9 5950x 16-core processor × 32
* OS (e.g., Linux): Ubuntu 22.04.2 LTS
* Python version: 3.9.16
* CUDA version: 11.6
* GPU models and configuration: GeForce RTX 3080 Ti

<details>
<summary>MobileNetV3 Large</summary>

| Inference time [ms/sample]    |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 4,392        | 2,515         | 3,838         | 4,055         |
| FP32 JIT CPU                  | 4,365        | 2,523         | 2,599         | 3,455         |
| INT8 CPU Dynamic Quantization | 6,077        | 2,492         | 2,693         | 3,98          |
| FP32 CUDA                     | 4,893        | 0,289         | 0,195         | 0,185         |
| FP32 JIT CUDA                 | 2,517        | 0,25          | 0,226         | 0,21          |
| FP16 CUDA                     | 4,116        | 0,528         | 0,417         | 0,368         |
| FP16 JIT CUDA                 | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |
| FP32 TensorRT                 | 0,707        | 0,125         | 0,117         | 0,109         |
| FP32 JIT TensorRT             | 1,907        | 0,181         | 0,153         | 0,151         |
| FP16 TensorRT                 | 0,536        | 0,074         | 0,065         | 0,06          |
| FP16 JIT TensorRT             | 1,949        | 0,158         | 0,132         | 0,123         |
| INT8 Quantized TensorRT       | 0,503        | 0,065         | 0,047         | 0,039         |


| GPU Memory Peak usage [MB] - max_memory_allocated |              |               |               |               |
|---------------------------------------------------|--------------|---------------|---------------|---------------|
|                                                   | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                                          | 0            | 0             | 0             | 0             |
| FP32 JIT CPU                                      | 0            | 0             | 0             | 0             |
| INT8 CPU Dynamic Quantization                     | 0            | 0             | 0             | 0             |
| FP32 CUDA                                         | 2419         | 2435          | 2509          | 2729          |
| FP32 JIT CUDA                                     | 2336         | 2399          | 2475          | 2646          |
| FP16 CUDA                                         | 1170         | 1219          | 1272          | 1380          |
| FP16 JIT CUDA                                     | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |
| FP32 TensorRT                                     | 2270         | 2275          | 2274          | 2293          |
| FP32 JIT TensorRT                                 | 2336         | 2399          | 2446          | 2544          |
| FP16 TensorRT                                     | 2270         | 2275          | 2274          | 2293          |
| FP16 JIT TensorRT                                 | 2336         | 2436          | 2438          | 2544          |
| INT8 Quantized TensorRT                           | 2270         | 2276          | 2285          | 2304          |

| F1 score                      |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 0,734        | 0,734         | 0,734         | 0,734         |
| FP32 JIT CPU                  | 0,734        | 0,734         | 0,734         | 0,734         |
| INT8 CPU Dynamic Quantization | 0,734        | 0,734         | 0,734         | 0,734         |
| FP32 CUDA                     | 0,734        | 0,734         | 0,734         | 0,734         |
| FP32 JIT CUDA                 | 0,734        | 0,734         | 0,734         | 0,734         |
| FP16 CUDA                     | 0,736        | 0,736         | 0,735         | 0,735         |
| FP16 JIT CUDA                 | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |
| FP32 TensorRT                 | 0,734        | 0,734         | 0,734         | 0,734         |
| FP32 JIT TensorRT             | 0,734        | 0,734         | 0,734         | 0,734         |
| FP16 TensorRT                 | 0,735        | 0,735         | 0,735         | 0,735         |
| FP16 JIT TensorRT             | 0,734        | 0,734         | 0,735         | 0,735         |
| INT8 Quantized TensorRT       | 0,695        | 0,701         | 0,7           | 0,709         |

</details>


<details>
<summary>ResNet18</summary>

| Inference time [ms/sample]    |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 6,617        | 3,863         | 4,002         | 4,658         |
| FP32 JIT CPU                  | 3,757        | 3,442         | 3,369         | 3,73          |
| INT8 CPU Dynamic Quantization | 6,442        | 3,578         | 3,809         | 4,553         |
| FP32 CUDA                     | 2,128        | 0,255         | 0,239         | 0,212         |
| FP32 JIT CUDA                 | 1,422        | 0,229         | 0,191         | 0,167         |
| FP16 CUDA                     | 2,043        | 0,422         | 0,411         | 0,435         |
| FP16 JIT CUDA                 | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |
| FP32 TensorRT                 | 0,779        | 0,22          | 0,202         | 0,184         |
| FP32 JIT TensorRT             | 1,526        | 0,262         | 0,216         | 0,199         |
| FP16 TensorRT                 | 0,327        | 0,067         | 0,062         | 0,056         |
| FP16 JIT TensorRT             | 1,511        | 0,26          | 0,215         | 0,199         |
| INT8 Quantized TensorRT       | 0,255        | 0,042         | 0,032         | 0,028         |


| GPU Memory Peak usage [MB] - max_memory_allocated |              |               |               |               |
|---------------------------------------------------|--------------|---------------|---------------|---------------|
|                                                   | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                                          | 0            | 0             | 0             | 0             |
| FP32 JIT CPU                                      | 0            | 0             | 0             | 0             |
| INT8 CPU Dynamic Quantization                     | 0            | 0             | 0             | 0             |
| FP32 CUDA                                         | 2327         | 2410          | 2497          | 2693          |
| FP32 JIT CUDA                                     | 2336         | 2399          | 2475          | 2646          |
| FP16 CUDA                                         | 1208         | 1258          | 1315          | 1423          |
| FP16 JIT CUDA                                     | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |
| FP32 TensorRT                                     | 2270         | 2275          | 2274          | 2293          |
| FP32 JIT TensorRT                                 | 2336         | 2399          | 2475          | 2646          |
| FP16 TensorRT                                     | 2270         | 2275          | 2274          | 2293          |
| FP16 JIT TensorRT                                 | 2336         | 2399          | 2475          | 2646          |
| INT8 Quantized TensorRT                           | 2270         | 2275          | 2285          | 2304          |

| F1 score                      |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 0,69         | 0,69          | 0,69          | 0,69          |
| FP32 JIT CPU                  | 0,69         | 0,69          | 0,69          | 0,69          |
| INT8 CPU Dynamic Quantization | 0,69         | 0,69          | 0,69          | 0,69          |
| FP32 CUDA                     | 0,69         | 0,69          | 0,69          | 0,69          |
| FP32 JIT CUDA                 | 0,69         | 0,69          | 0,69          | 0,69          |
| FP16 CUDA                     | 0,69         | 0,69          | 0,69          | 0,69          |
| FP16 JIT CUDA                 | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |
| FP32 TensorRT                 | 0,69         | 0,69          | 0,69          | 0,69          |
| FP32 JIT TensorRT             | 0,69         | 0,69          | 0,69          | 0,69          |
| FP16 TensorRT                 | 0,69         | 0,69          | 0,69          | 0,69          |
| FP16 JIT TensorRT             | 0,69         | 0,69          | 0,69          | 0,69          |
| INT8 Quantized TensorRT       | 0,689        | 0,689         | 0,689         | 0,689         |

</details>


<details>
<summary>FCN</summary>

| Inference time [ms/sample]    |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 3,666        | 0,428         | 0,39          | 0,391         |
| FP32 JIT CPU                  | 7,619        | 0,691         | 0,58          | 0,56          |
| INT8 CPU Dynamic Quantization | 3,473        | 0,451         | 0,41          | 0,425         |
| FP32 CUDA                     | 0,24         | 0,018         | 0,011         | 0,009         |
| FP32 JIT CUDA                 | 0,22         | 0,017         | 0,015         | 0,013         |
| FP16 CUDA                     | 0,167        | 0,012         | 0,009         | 0,005         |
| FP16 JIT CUDA                 | 0,161        | 0,019         | 0,015         | 0,011         |
| FP32 TensorRT                 | 0,276        | 0,024         | 0,015         | 0,01          |
| FP32 JIT TensorRT             | 0,278        | 0,024         | 0,015         | 0,01          |
| FP16 TensorRT                 | 0,215        | 0,02          | 0,012         | 0,009         |
| FP16 JIT TensorRT             | 0,221        | 0,019         | 0,015         | 0,01          |
| INT8 Quantized TensorRT       | 0,286        | 0,027         | 0,016         | 0,011         |


| GPU Memory Peak usage [MB] - max_memory_allocated |              |               |               |               |
|---------------------------------------------------|--------------|---------------|---------------|---------------|
|                                                   | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                                          | 0            | 0             | 0             | 0             |
| FP32 JIT CPU                                      | 0            | 0             | 0             | 0             |
| INT8 CPU Dynamic Quantization                     | 0            | 0             | 0             | 0             |
| FP32 CUDA                                         | 2401         | 2407          | 2416          | 2435          |
| FP32 JIT CUDA                                     | 2530         | 2545          | 2563          | 2595          |
| FP16 CUDA                                         | 1332         | 1338          | 1347          | 1366          |
| FP16 JIT CUDA                                     | 2595         | 2606          | 2619          | 2647          |
| FP32 TensorRT                                     | 2400         | 2406          | 2415          | 2433          |
| FP32 JIT TensorRT                                 | 2530         | 2536          | 2545          | 2563          |
| FP16 TensorRT                                     | 2270         | 2276          | 2285          | 2304          |
| FP16 JIT TensorRT                                 | 2530         | 2536          | 2545          | 2563          |
| INT8 Quantized TensorRT                           | 2270         | 2276          | 2285          | 2304          |

</details>

<details>
<summary>CNN</summary>

| Inference time [ms/sample]    |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 0,63         | 0,485         | 0,542         | 0,645         |
| FP32 JIT CPU                  | 0,7          | 0,718         | 0,748         | 0,852         |
| INT8 CPU Dynamic Quantization | 0,585        | 0,401         | 0,452         | 0,543         |
| FP32 CUDA                     | 0,368        | 0,067         | 0,06          | 0,057         |
| FP32 JIT CUDA                 | 0,296        | 0,055         | 0,046         | 0,053         |
| FP16 CUDA                     | 0,368        | 0,104         | 0,094         | 0,096         |
| FP16 JIT CUDA                 | 0,296        | 0,058         | 0,047         | 0,054         |
| FP32 TensorRT                 | 0,167        | 0,031         | 0,026         | 0,024         |
| FP32 JIT TensorRT             | 0,298        | 0,057         | 0,048         | 0,046         |
| FP16 TensorRT                 | 0,155        | 0,023         | 0,018         | 0,016         |
| FP16 JIT TensorRT             | 0,3          | 0,057         | 0,049         | 0,045         |
| INT8 Quantized TensorRT       | 0,161        | 0,024         | 0,02          | 0,018         |

| GPU Memory Peak usage [MB] - max_memory_allocated |              |               |               |               |
|---------------------------------------------------|--------------|---------------|---------------|---------------|
|                                                   | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                                          | 0            | 0             | 0             | 0             |
| FP32 JIT CPU                                      | 0            | 0             | 0             | 0             |
| INT8 CPU Dynamic Quantization                     | 0            | 0             | 0             | 0             |
| FP32 CUDA                                         | 2272         | 2304          | 2340          | 2409          |
| FP32 JIT CUDA                                     | 2272         | 2300          | 2331          | 2393          |
| FP16 CUDA                                         | 1144         | 1159          | 1177          | 1215          |
| FP16 JIT CUDA                                     | 2273         | 2300          | 2331          | 2393          |
| FP32 TensorRT                                     | 2271         | 2277          | 2286          | 2304          |
| FP32 JIT TensorRT                                 | 2272         | 2300          | 2331          | 2393          |
| FP16 TensorRT                                     | 2270         | 2276          | 2285          | 2304          |
| FP16 JIT TensorRT                                 | 2272         | 2300          | 2331          | 2393          |
| INT8 Quantized TensorRT                           | 2270         | 2276          | 2285          | 2304          |

</details>

<details>
<summary>LSTM</summary>

| Inference time [ms/sample]    |              |               |               |               |
|-------------------------------|--------------|---------------|---------------|---------------|
|                               | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                      | 11,007       | 5,434         | 2,941         | 1,653         |
| FP32 JIT CPU                  | 11,015       | 5,336         | 2,876         | 1,57          |
| INT8 CPU Dynamic Quantization | 11,587       | 5,314         | 2,849         | 1,443         |
| FP32 CUDA                     | 3,343        | 0,229         | 0,114         | 0,061         |
| FP32 JIT CUDA                 | 4,96         | 0,33          | 0,168         | 0,091         |
| FP16 CUDA                     | 4,459        | 0,278         | 0,142         | 0,078         |
| FP16 JIT CUDA                 | 4,407        | 0,277         | 0,134         | 0,079         |
| FP32 TensorRT                 | 3,398        | 0,234         | 0,124         | 0,068         |
| FP32 JIT TensorRT             | 5,043        | 0,339         | 0,177         | 0,096         |
| FP16 TensorRT                 | 3,375        | 0,245         | 0,124         | 0,069         |
| FP16 JIT TensorRT             | 5,069        | 0,354         | 0,178         | 0,096         |
| INT8 Quantized TensorRT       | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |

| GPU Memory Peak usage [MB] - max_memory_allocated |              |               |               |               |
|---------------------------------------------------|--------------|---------------|---------------|---------------|
|                                                   | Batch size 1 | Batch size 16 | Batch size 32 | Batch size 64 |
| FP32 CPU                                          | 0            | 0             | 0             | 0             |
| FP32 JIT CPU                                      | 0            | 0             | 0             | 0             |
| INT8 CPU Dynamic Quantization                     | 0            | 0             | 0             | 0             |
| FP32 CUDA                                         | 2487         | 2504          | 2519          | 2565          |
| FP32 JIT CUDA                                     | 2594         | 2611          | 2626          | 2671          |
| FP16 CUDA                                         | 1352         | 1362          | 1370          | 1397          |
| FP16 JIT CUDA                                     | 2486         | 2499          | 2507          | 2540          |
| FP32 TensorRT                                     | 2487         | 2516          | 2549          | 2613          |
| FP32 JIT TensorRT                                 | 2594         | 2623          | 2655          | 2719          |
| FP16 TensorRT                                     | 2487         | 2516          | 2549          | 2613          |
| FP16 JIT TensorRT                                 | 2594         | 2623          | 2655          | 2719          |
| INT8 Quantized TensorRT                           | RuntimeError | RuntimeError  | RuntimeError  | RuntimeError  |

</details>
