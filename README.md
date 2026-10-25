# Efficient Deep Learning: Model Compression, Quantization, and Optimization

This research-oriented project investigates model compression techniques for deep learning systems operating under resource constraints.

## Overview

Deep neural networks can require considerable storage and computational resources. This project examines whether post-training quantization and model pruning can reduce model size and improve inference efficiency while maintaining classification performance.

The experiments use **ResNet-18 on CIFAR-100**, comparing a full-precision baseline with compressed model variants.

## Research Questions

- To what extent can INT8 quantization reduce model checkpoint size?
- How does compression affect classification performance?
- Does combining pruning with quantization provide additional benefits?
- How do the evaluated methods compare in terms of CPU inference performance?

## Methods

Three model configurations are evaluated:

1. **FP32 baseline:** The full-precision model.
2. **Static INT8 quantization:** A quantization approach that uses reduced precision for model weights and operations through a static quantization workflow.
3. **20% pruning + static INT8 quantization:** A combined approach that applies pruning and quantization.

The experiments compare classification metrics, checkpoint size, evaluation time, and throughput.

## Experimental Results

| Metric | FP32 | Static INT8 | 20% Pruning + INT8 |
|---|---:|---:|---:|
| Test accuracy | 38.94% | 38.66% | 38.71% |
| Macro-F1 | 0.3788 | 0.3759 | 0.3765 |
| Evaluation time | 79.95 s | 21.49 s | 21.27 s |
| Throughput | 125.08 images/s | 465.29 images/s | 470.11 images/s |
| Checkpoint size | 44.96 MB | 11.36 MB | 11.36 MB |

*These results come from the project's recorded evaluation run. Timing and throughput depend on the hardware and evaluation pipeline.*

## Key Findings

- Static INT8 quantization reduced checkpoint size by approximately **74.7%** compared with the FP32 baseline.
- Test accuracy decreased by **0.28 percentage points** following quantization.
- The recorded throughput of the INT8 model was approximately **3.72 times higher** than that of the FP32 model on the tested CPU system.
- Combining 20% pruning with INT8 quantization did not lead to a meaningful further reduction in checkpoint size or a substantial improvement in throughput compared with INT8 quantization alone.

These findings are specific to the evaluated model, dataset, implementation, and hardware configuration. They should not be generalized to other architectures or deployment environments without further experiments.

## Evaluation

Classification performance is assessed using test accuracy and macro-F1. Checkpoint size is used to measure storage requirements, while evaluation time and throughput are used to compare inference performance.

## Project Structure

```text
AI_Model_Compression/
├── figures/    # Experimental plots and visualizations
├── reports/    # Research reports and analysis
├── src/        # Source code
├── .gitignore
└── README.md
```

## Reproducibility

The project was developed using Python 3.11.9 in a CPU-only Windows environment with 8 GB of RAM.

The source code and reports provide the basis for examining the dependencies, data preparation, calibration procedure, model checkpoints, and evaluation commands. Before running the experiments, verify the relevant commands against the current repository contents.

## Limitations

- The experiments are limited to ResNet-18 and CIFAR-100.
- The results depend on the tested CPU environment and evaluation pipeline.
- The recorded experiments do not establish that pruning consistently improves model size or inference speed.
- Additional experiments with other architectures and deployment environments are required to evaluate how well these findings generalize.

## Future Work

Future work could examine other quantization strategies, evaluate additional architectures, investigate structured pruning, and measure deployment performance on resource-constrained devices.

## Author

**Poorchista Jafari**

GitHub: [PoorchistaJafari](https://github.com/PoorchistaJafari)
