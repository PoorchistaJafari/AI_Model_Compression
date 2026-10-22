# Research Plan: Model Compression and Quantization

## 1. Project Title

Efficient Deep Learning: Model Compression, Quantization, and Optimization for Resource-Constrained AI Systems

## 2. Research Question

Under a controlled image-classification setting, how do different compression strategies affect the trade-off among predictive performance, model size, and inference efficiency?

## 3. Research Objective

The objective is to evaluate how structured channel pruning and INT8 post-training quantization affect a deep learning model's predictive performance, storage requirements, and inference efficiency.

The project will use controlled experiments and quantitative comparisons rather than reporting accuracy alone.

## 4. Dataset and Model

- Dataset: CIFAR-100
- Task: Image classification
- Model: CIFAR-adapted ResNet-18
- Computing environment: CPU-only Windows laptop
- Available RAM: 8 GB
- NVIDIA GPU: Not available

## 5. Experimental Conditions

1. Baseline model using FP32 weights
2. Structured pruning at 20%
3. Structured pruning at 40%
4. Structured pruning at 60%
5. INT8 post-training quantization
6. 20% pruning followed by INT8 quantization
7. 40% pruning followed by INT8 quantization
8. 60% pruning followed by INT8 quantization

The feasibility and implementation details of each condition will be verified before running the full experiments.

## 6. Evaluation Metrics

- Top-1 classification accuracy
- Macro-F1 score
- Per-class performance and confusion matrix
- Number of model parameters
- Saved model file size
- Estimated computational cost (FLOPs)
- CPU inference latency
- Throughput
- Peak memory usage, where measurable

## 7. Experimental Methodology

1. Inspect and prepare the dataset.
2. Implement and evaluate the baseline model.
3. Apply the selected compression techniques.
4. Evaluate each model under consistent conditions.
5. Record results in structured tables.
6. Compare accuracy, model size, and inference efficiency.
7. Analyze errors, limitations, and trade-offs.

## 8. Expected Research Contribution

The project aims to provide an evidence-based comparison of compression strategies and identify which approaches offer a useful balance between predictive performance and computational efficiency under resource constraints.

No outcome is assumed in advance; conclusions will be based on experimental results.

## 9. Limitations

Experiments will be conducted on CPU-only hardware with 8 GB RAM. Training time, available memory, and software compatibility may limit the number or scale of experiments. Any adjustments will be documented transparently.

## 10. Reproducibility

The project will maintain organized source code, configuration files, experimental records, evaluation results, and documentation. Random seeds and relevant environment details will be recorded where applicable.

## Dataset Inspection Results

### Dataset

CIFAR-100

### Training Set

- Total images: 50,000
- Number of classes: 100
- Images per class: 500

### Test Set

- Total images: 10,000

### Initial Findings

The training dataset is balanced in terms of the number of images per class. Each of the 100 classes contains exactly 500 training images.

This balance reduces concerns about class-frequency imbalance during training. However, equal class counts do not guarantee equal classification difficulty across classes.

### Class Distribution: Training and Test Sets

The CIFAR-100 training set contains 50,000 images, with 500 images per class. The test set contains 10,000 images, with 100 images per class.

Both sets are balanced in terms of class frequency. The training-to-test split is 5:1.

Equal class frequencies do not necessarily imply equal classification difficulty across classes.

## Pixel Statistics

Pixel statistics were calculated using the CIFAR-100 training set after converting images to tensors with pixel values in the range [0, 1].

| RGB Channel |   Mean | Standard Deviation |
| ----------- | -----: | -----------------: |
| Red         | 0.5071 |             0.2673 |
| Green       | 0.4866 |             0.2564 |
| Blue        | 0.4409 |             0.2762 |

These statistics can be used to normalize the input images during model training. The mean and standard deviation were calculated from the training set only to avoid using test-set information during preprocessing.

## DataLoader Preparation

The CIFAR-100 training and test datasets were successfully loaded using PyTorch DataLoader.

- Training samples: 50,000
- Test samples: 10,000
- Batch size: 128
- Training batches: 391
- Test batches: 79
- Input tensor shape: [128, 3, 32, 32]
- Label tensor shape: [128]

Images were converted to tensors and normalized using the training-set RGB statistics. The final batch in each dataset is smaller than the configured batch size because the dataset sizes are not divisible by 128.

The data pipeline is ready for model training.

## Baseline Model: ResNet-18

A ResNet-18 model was created using PyTorch and adapted for CIFAR-100 classification.

### Architecture Configuration

- Initial convolution: 3 × 3 kernel, stride 1, padding 1
- Initial max-pooling: removed
- Output classes: 100
- Pretrained weights: not used

### Initial Model Validation

- Input shape: [1, 3, 32, 32]
- Output shape: [1, 100]
- Total parameters: 11,220,132
- Trainable parameters: 11,220,132

The model successfully processed a dummy input and produced 100 output scores, confirming that its architecture is compatible with the CIFAR-100 input dimensions.

This was an architecture validation test only. The model has not yet been trained, and its classification accuracy has not been measured.

## Baseline Model Size

The baseline ResNet-18 model was analyzed before training.

- Total parameters: 11,220,132
- Trainable parameters: 11,220,132
- Parameter storage: 42.80 MiB
- Buffer storage: 0.04 MiB
- Total model state: 42.84 MiB
- Parameter data type: FP32

The parameter storage estimate is based on the number of parameters and their data type. Actual checkpoint size and runtime memory usage may differ.

These measurements establish the reference point for evaluating model compression techniques.

## Baseline CPU Inference Benchmark

An initial CPU inference benchmark was conducted using the untrained ResNet-18 model.

- Device: CPU
- CPU threads: 2
- Batch size: 1
- Warm-up runs: 10
- Benchmark runs: 50
- Average latency: 15.09 ms/image
- Throughput: 66.25 images/second

The benchmark used a synthetic input tensor and measured inference without gradient computation. These values provide an initial reference only; final comparisons should use the same hardware, batch size, thread settings, and measurement procedure for all model variants.

## Initial Training Pipeline Test

A short training run was conducted to validate the training pipeline on a CPU-only system.

- Architecture: ResNet-18
- Epochs configured: 1
- Training batches processed: 20
- Training images processed: 2,560
- Training loss: 4.6198
- Training accuracy on the processed subset: 1.84%
- Test images evaluated: 1,280
- Accuracy on the evaluated test subset: 2.66%
- Training time: 1.29 minutes
- Saved weights: models/resnet18_baseline.pth

This was a limited pipeline validation experiment, not a completed baseline training run. The test accuracy is based on only a subset of the test set and must not be reported as final model performance.

## Training and Validation Pipeline Test

### Configuration

- Model: ResNet-18 adapted for CIFAR-100
- Device: CPU
- CPU threads: 2
- Optimizer: SGD
- Learning rate: 0.01
- Momentum: 0.9
- Weight decay: 0.0005
- Training batches: 20
- Batch size: 128
- Epochs: 1

### Data Split

- Training subset: 45,000 images
- Validation subset: 5,000 images
- Test set: 10,000 images, reserved for final evaluation

### Initial Results

- Training images processed: 2,560
- Average training loss: 4.6281
- Training accuracy: 1.84%
- Training time: 1.28 minutes
- Validation images evaluated: 1,280
- Validation accuracy: 2.73%

### Interpretation

This experiment verifies that the training and validation pipeline runs successfully on CPU. The model was trained on a limited subset of the training data and evaluated on a separate subset of the validation data. These results are preliminary and must not be interpreted as the final baseline performance.

### Saved Artifact

- Model weights: `models/resnet18_baseline.pth`

### Limitations

- Only 20 training batches were processed.
- Validation accuracy was calculated using 1,280 of the 5,000 validation images.
- The model has not undergone full training.
- The test set has not been used for final evaluation.

### Re-run After Data Pipeline Refinement

The training and validation pipeline was rerun successfully after separating the dataset objects and fixing the reproducible split indices.

- Training images processed: 2,560
- Average training loss: 4.6281
- Training accuracy: 1.84%
- Training time: 1.08 minutes
- Validation images evaluated: 1,280
- Validation accuracy: 2.73%
- Test set: Not used
- Model weights: Saved successfully to `models/resnet18_baseline.pth`

The experiment confirms that the updated pipeline runs successfully. The results remain preliminary because training and validation were limited to a subset of the available data.

## Baseline Model Checkpoint: Initial Size

- File: `models/resnet18_baseline.pth`
- File size: 44,959,307 bytes
- Approximate decimal size: 44.96 MB
- Approximate binary size: 42.88 MiB
- Checkpoint status: Saved successfully
- Training status: Preliminary pipeline-test weights; not a fully trained baseline

This checkpoint provides an initial on-disk size reference. Final compression comparisons must use a sufficiently trained baseline and consistent measurement procedures.

## Baseline Checkpoint Loading Test

- Checkpoint: `models/resnet18_baseline.pth`
- Loading status: Successful
- Device: CPU
- Test input shape: `(1, 3, 32, 32)`
- Model output shape: `(1, 100)`
- All output values finite: True

The saved checkpoint was successfully loaded into the ResNet-18 architecture, and inference produced a valid output tensor for 100 classes. This confirms checkpoint compatibility and basic inference functionality, but does not establish classification quality or full-training completion.

## CPU Training Cost Estimate

### Benchmark Configuration

- Device: CPU
- PyTorch threads: 2
- Batch size: 128
- Model: ResNet-18 adapted for CIFAR-100
- One training batch (forward pass, loss computation, backward pass, and optimizer update): 3.13 seconds

### Estimated Training Time

- 20 batches: approximately 1.04 minutes
- 100 batches: approximately 5.22 minutes
- 391 batches: approximately 20.42 minutes

### Interpretation and Limitations

These estimates assume a roughly constant batch-processing time. Actual training time may vary because of data loading, batch size, system load, and other overhead. A full training epoch is estimated to take around 20–22 minutes, excluding additional validation and checkpoint-saving time.

This benchmark will guide the choice of a practical CPU training schedule. The final baseline must be trained sufficiently to support meaningful comparisons with pruning and quantization methods.

## Baseline Training Results — 3 Epochs

### Configuration

- Model: ResNet-18 adapted for CIFAR-100
- Device: CPU
- CPU threads: 2
- Training subset: 45,000 images
- Validation subset: 5,000 images
- Batch size: 128
- Epochs: 3
- Optimizer: SGD
- Learning rate: 0.01
- Momentum: 0.9
- Weight decay: 0.0005

### Results by Epoch

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |      Time |
| ----- | ------------: | ----------------: | --------------: | ------------------: | --------: |
| 1     |        3.7656 |            12.47% |          3.4262 |              19.26% | 20.99 min |
| 2     |        2.9679 |            26.02% |          2.9161 |              26.74% | 23.22 min |
| 3     |        2.4344 |            36.28% |          2.5665 |              34.50% | 33.54 min |

### Interpretation

Training accuracy increased from 12.47% to 36.28%, while validation accuracy increased from 19.26% to 34.50%. Validation loss decreased throughout the three epochs, indicating continued improvement during this training run.

The best validation accuracy observed was 34.50% at epoch 3. This is an initial trained baseline, not a final benchmark. The test set has not been evaluated.

### Checkpoint

- File: `models/resnet18_baseline.pth`
- The checkpoint is saved whenever validation accuracy improves.
- The checkpoint currently corresponds to the best validation result observed in this run.

### Limitations

- Only three epochs were trained.
- The test set remains unused.
- Further training or hyperparameter tuning may improve performance.
- Compression experiments should use the same trained checkpoint and consistent evaluation procedures.

## Baseline Evaluation — Full Validation Set

### Evaluation Setup

- Model: ResNet-18 adapted for CIFAR-100
- Device: CPU
- CPU threads: 2
- Checkpoint: `models/resnet18_baseline.pth`
- Validation images: 5,000
- Batch size: 128

### Results

- Validation loss: 2.5665
- Validation accuracy: 34.50%
- Validation Macro-F1: 0.3177
- Evaluation time: 37.47 seconds
- Throughput: 133.43 images/second

### Interpretation

The saved baseline checkpoint was evaluated on the complete validation subset. Accuracy and Macro-F1 provide complementary measures of classification performance. The observed throughput is an initial CPU measurement under the current batch configuration and should not be treated as a definitive single-image latency benchmark.

### Evaluation Protocol

The test set remains reserved for final evaluation. Compression methods should be compared against this same trained checkpoint, using the same validation data and consistent measurement settings.

## Baseline Model Size and Parameter Analysis

### Model Configuration

- Architecture: ResNet-18 adapted for CIFAR-100
- Input image size: 32 × 32
- Number of output classes: 100
- Hardware: CPU-only system

### Model Complexity and Storage

| Metric                  | Baseline Result |
| ----------------------- | --------------: |
| Total parameters        |      11,220,132 |
| Trainable parameters    |      11,220,132 |
| Checkpoint size (bytes) |      44,959,307 |
| Checkpoint size (MB)    |           44.96 |
| Checkpoint size (MiB)   |           42.88 |

### Baseline Evaluation Reference

| Metric                        |               Result |
| ----------------------------- | -------------------: |
| Validation accuracy           |               34.50% |
| Validation Macro-F1           |               0.3177 |
| Validation loss               |               2.5665 |
| Initial evaluation throughput | 133.43 images/second |

### Interpretation

The baseline ResNet-18 contains approximately 11.22 million parameters and occupies 44.96 MB on disk. These measurements establish the reference point for evaluating pruning and quantization.

Future experiments will compare model size, parameter count, predictive performance, and inference speed against this baseline. The initial evaluation throughput is a reference measurement rather than a fully controlled latency benchmark.

The validation set is used for model selection and development. The test set remains reserved for final evaluation after the compression experiments and evaluation procedure have been finalized.
