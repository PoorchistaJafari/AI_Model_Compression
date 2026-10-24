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

## Initial Pruning Experiment — 20% Global Magnitude Pruning

### Method

Global unstructured magnitude pruning was applied to the weights of convolutional layers and the final fully connected classifier. The 20% pruning target was applied across the selected weights. Pruning was then made permanent using PyTorch's pruning removal utility.

### Validation Results

| Metric              | Baseline | Pruned Model |
| ------------------- | -------: | -----------: |
| Validation accuracy |   34.50% |       34.64% |
| Validation Macro-F1 |   0.3177 |       0.3196 |
| Target sparsity     |       0% |       20.00% |

### Initial Interpretation

The pruned model achieved 34.64% validation accuracy and a Macro-F1 score of 0.3196. Compared with the baseline, accuracy increased by 0.14 percentage points and Macro-F1 increased by 0.0019.

These small differences should not be interpreted as conclusive evidence that pruning improves predictive performance. The experiment is preliminary and requires additional measurements, including checkpoint size and inference performance.

The measured 20% sparsity applies only to the selected convolutional and classifier weights. It does not mean that the entire model has 20% of its parameters removed or that the saved checkpoint will be 20% smaller.

## Pruning Experiment — 20% Global Magnitude Pruning

### Method

Global unstructured magnitude pruning was applied to the weights of all convolutional layers and the final fully connected classifier. The target pruning amount was 20% of the selected weights. The pruning masks were then made permanent using PyTorch's pruning removal utility, and the resulting model weights were saved to a separate checkpoint.

### Results

| Metric                     |   Baseline | Pruned Model |
| -------------------------- | ---------: | -----------: |
| Validation accuracy        |     34.50% |       34.64% |
| Validation Macro-F1        |     0.3177 |       0.3196 |
| Validation loss            |     2.5665 |       2.5577 |
| Selected-weight sparsity   |         0% |       20.00% |
| Checkpoint size (bytes)    | 44,959,307 |   44,959,435 |
| Checkpoint size (MB)       |      44.96 |        44.96 |
| Evaluation time (seconds)  |      37.47 |        35.13 |
| Throughput (images/second) |     133.43 |       142.33 |

### Interpretation and Limitations

The pruning procedure successfully set 2,242,086 of 11,210,432 selected weights to zero, achieving 20.00% sparsity among the selected convolutional and classifier weights.

The pruned model achieved 34.64% validation accuracy, compared with 34.50% for the baseline. This small difference does not establish that pruning improves generalization.

The checkpoint size increased by 128 bytes rather than decreasing. This is expected because the pruned weights were saved in a dense tensor representation, where zero-valued weights still occupy storage.

The observed throughput was higher for the pruned model in this single evaluation run. However, this measurement is not sufficient to establish a reliable inference speed improvement. Repeated, controlled benchmarking and an appropriate sparse or compressed representation are required to evaluate practical efficiency gains.

The current experiment demonstrates weight sparsification, but it does not yet demonstrate a reduction in dense checkpoint size or a verified speedup.

### Saved Artifact

- Checkpoint: `models/resnet18_pruned_20.pth`

## Pruning Experiment — 40% Global Magnitude Pruning

### Method

Global unstructured L1 magnitude pruning was applied to the
convolutional-layer weights and final fully connected layer weights.
The pruning experiment started from the original trained baseline
checkpoint to ensure a fair comparison with the 20% pruning experiment.

### Results

| Metric               |               Result |
| -------------------- | -------------------: |
| Validation images    |                5,000 |
| Target pruning ratio |                  40% |
| Selected weights     |           11,210,432 |
| Zero weights         |            4,484,173 |
| Measured sparsity    |               40.00% |
| Validation loss      |               2.5371 |
| Validation accuracy  |               34.36% |
| Validation Macro-F1  |               0.3219 |
| Evaluation time      |        34.92 seconds |
| Throughput           | 143.19 images/second |
| Checkpoint size      |             44.96 MB |

### Interpretation

The model achieved 40.00% sparsity among the selected weights.
Validation accuracy was 0.14 percentage points lower than the
baseline, while Macro-F1 increased from 0.3177 to 0.3219 and
validation loss decreased from 2.5665 to 2.5371.

The checkpoint size remained approximately 44.96 MB because the
zero-valued weights were stored in the standard dense format.
The observed evaluation throughput should not be interpreted as
proof of a speedup because the timing measurements have not yet
been repeated under controlled conditions.

These results are preliminary and are based on the validation set.
The test set remains reserved for final evaluation.

## Pruning Results Visualization

The figure `figures/pruning_comparison.png` compares baseline,
20% global magnitude pruning, and 40% global magnitude pruning
using validation accuracy, Macro-F1, and checkpoint size.

The results show that validation accuracy remained close to the
baseline across both pruning levels. Macro-F1 increased slightly
in these individual runs, but repeated experiments are needed
before drawing firm conclusions.

Checkpoint sizes remained approximately 44.96 MB because pruning
produced zero-valued weights without changing the dense storage
format. Therefore, sparsity alone did not reduce the saved
checkpoint size.

## Quantization Experiment — Dynamic INT8

### Method

Dynamic INT8 quantization was applied to the supported Linear
layer of the trained ResNet-18 model using PyTorch.
The original FP32 baseline checkpoint was used as the starting point.

Only the final fully connected layer was quantized. Convolutional
layers remained in floating-point format.

### Results

| Metric              |   FP32 Baseline |    Dynamic INT8 |
| ------------------- | --------------: | --------------: |
| Validation accuracy |          34.50% |          34.48% |
| Validation Macro-F1 |          0.3177 |          0.3175 |
| Validation loss     |          2.5665 |          2.5653 |
| Evaluation time     |         37.47 s |         34.14 s |
| Throughput          | 133.43 images/s | 146.47 images/s |
| Checkpoint size     |        44.96 MB |        44.81 MB |

Checkpoint size decreased by approximately 0.34%, while validation
accuracy decreased by 0.02 percentage points.

### Interpretation and Limitations

The small reduction in checkpoint size is expected because only
the final Linear layer was quantized; most model parameters belong
to convolutional layers that remained in FP32.

The single-run timing results are preliminary and do not establish
a reliable speedup. Repeated, controlled benchmarking is required.

This experiment demonstrates that dynamic quantization of the final
classifier alone is insufficient for substantial compression of
this ResNet-18 model. The test set remains reserved for final evaluation.

## Static INT8 Quantization Experiment

### Method

FX graph-mode static INT8 quantization was applied to the trained
ResNet-18 model using the oneDNN quantization backend.

The model was calibrated using 1,024 training images and evaluated
on the 5,000-image validation set. The original FP32 checkpoint was
used as the baseline.

### Results

| Metric                     |   FP32 Baseline |     Static INT8 |
| -------------------------- | --------------: | --------------: |
| Validation loss            |          2.5665 |          2.5673 |
| Validation accuracy        |          34.50% |          34.36% |
| Validation Macro-F1        |          0.3177 |          0.3158 |
| Evaluation time            |         37.47 s |          8.85 s |
| Throughput                 | 133.43 images/s | 564.89 images/s |
| Serialized state-dict size |        44.96 MB |        11.35 MB |

The serialized state-dict size decreased by approximately 74.75%.
Validation accuracy decreased by 0.14 percentage points.

### Interpretation and Limitations

Static INT8 quantization converted supported convolutional and
linear operations to quantized operations. The measured validation
accuracy remained close to the FP32 baseline, while the serialized
weight file became substantially smaller.

The observed timing improvement is preliminary because each model
was measured in a single run. Repeated controlled benchmarks are
needed for a reliable performance comparison.

The reported quantized file size refers to the serialized state
dictionary, not necessarily a complete standalone deployment artifact.
The test set remains reserved for final evaluation.

## Combined Experiment — 20% Pruning + Static INT8 Quantization

### Method

The experiment started from the trained FP32 baseline.
Global unstructured L1 magnitude pruning was applied at a target
sparsity of 20% to convolutional weights and the final classifier
weights. Static FX graph-mode INT8 quantization was then applied
using the oneDNN backend.

Calibration used 1,024 training images. Evaluation used the
5,000-image validation set.

### Results

| Metric                     |   FP32 Baseline |     Static INT8 | Pruning 20% + Static INT8 |
| -------------------------- | --------------: | --------------: | ------------------------: |
| Validation accuracy        |          34.50% |          34.36% |                    34.48% |
| Validation Macro-F1        |          0.3177 |          0.3158 |                    0.3187 |
| Validation loss            |          2.5665 |          2.5673 |                    2.5577 |
| Evaluation time            |         37.47 s |          8.85 s |                    9.06 s |
| Throughput                 | 133.43 images/s | 564.89 images/s |           552.16 images/s |
| Serialized state-dict size |        44.96 MB |        11.35 MB |                  11.36 MB |

The combined model achieved 20.00% sparsity among the selected
weights and a measured state-dict size reduction of 74.74%
relative to the FP32 baseline.

### Interpretation and Limitations

The combined model retained validation accuracy close to the
baseline. However, its serialized state dictionary was slightly
larger than that of the Static INT8 model alone. Thus, this
experiment did not demonstrate an additional storage benefit
from applying 20% pruning before quantization.

Timing results are preliminary because each model was measured
in a single run. Repeated controlled benchmarks are required.
The reported file sizes refer to serialized state dictionaries,
not necessarily complete standalone deployment artifacts.
The test set remains reserved for final evaluation.

## Overall Compression Comparison

The figure `figures/compression_comparison.png` compares the
baseline FP32 model, two pruning configurations, dynamic INT8,
static INT8, and the combined 20% pruning plus static INT8 model.

Across these experiments, validation accuracy remained close to
34.5%. Static INT8 quantization reduced the serialized state-dict
size from approximately 44.96 MB to 11.35 MB, a reduction of
about 74.75%.

Combining 20% pruning with static INT8 resulted in a state-dict
size of approximately 11.36 MB. This did not provide an additional
storage reduction compared with static INT8 alone.

These findings are preliminary and based on individual validation
runs. Repeated benchmarking and final evaluation on the held-out
test set are still required.

## Model Serialization and Reload Test

### Objective

Verify that the static INT8 model can be reconstructed, loaded from its saved state dictionary, and used for inference.

### Results

- PyTorch version: 2.14.1+cpu
- Quantized state_dict size: 11,353,661 bytes
- Full FX model test file size: 11,358,655 bytes
- Output shape: (2, 100)
- Maximum output difference after reconstruction and state_dict loading: 0.00000000
- Output comparison: PASSED

### Interpretation

The quantized model was successfully reconstructed, and its saved state dictionary was loaded into the reconstructed graph. The outputs matched exactly for the tested random input batch.

The full-model serialization experiment using torch.save and torch.load(weights_only=False) failed during deserialization in the current PyTorch environment. Therefore, full-object serialization and deserialization have not been validated successfully. The successful state_dict reconstruction test provides a reproducible alternative.

### Limitations

The reload test used random calibration inputs and a random inference batch. It validates serialization and output consistency for this test, not model accuracy on CIFAR-100. The full-model test file is an experimental artifact, not a confirmed deployment package.

## Extended Baseline Training — Two Additional Epochs

### Objective

Investigate whether additional training improves the validation performance of the existing ResNet-18 baseline on CIFAR-100.

### Method

The model weights from the original three-epoch baseline checkpoint were loaded, and training continued for two additional epochs. A new SGD optimizer was initialized; the optimizer state from the original run was not available.

The original checkpoint was preserved. The best model from the additional training run was saved separately as `models/resnet18_baseline_extended.pth`, based on validation accuracy.

### Results

| Training stage             | Training accuracy | Validation loss | Validation accuracy |
| -------------------------- | ----------------: | --------------: | ------------------: |
| Original baseline, epoch 3 |            36.28% |          2.5665 |              34.50% |
| Additional epoch 1         |            45.25% |          2.3520 |              39.00% |
| Additional epoch 2         |            53.74% |          2.3444 |              38.84% |

Additional training time:

- Epoch 1: 21.08 minutes
- Epoch 2: 27.17 minutes

### Interpretation

The best validation accuracy increased from 34.50% to 39.00%, an improvement of 4.50 percentage points. Validation loss also decreased compared with the original baseline.

Training accuracy continued to increase during the second additional epoch, while validation accuracy decreased slightly from 39.00% to 38.84%. This may indicate the beginning of overfitting, but two additional epochs are insufficient to establish that conclusion.

### Limitations

The results are based on the validation subset. The held-out test set has not been used. Further controlled experiments are needed before drawing conclusions about generalization or treating this model as the final baseline for compression comparisons.

## Extended Baseline — 20% Global Magnitude Pruning

### Objective

Evaluate the effect of applying 20% global unstructured magnitude pruning to the extended ResNet-18 baseline.

### Method

The extended baseline checkpoint, `models/resnet18_baseline_extended.pth`, was loaded. Global L1 unstructured pruning was applied to the convolutional weights and final fully connected layer weights. Pruning was then made permanent by removing the pruning reparameterization.

The pruned model was evaluated on the 5,000-image validation subset and saved as `models/resnet18_extended_pruned_20.pth`.

### Results

- Target pruning ratio: 20%
- Selected weights: 11,210,432
- Zero weights after pruning: 2,242,086
- Measured sparsity: 20.00%
- Validation loss: 2.3509
- Validation accuracy: 38.84%
- Validation Macro-F1: 0.3764
- Evaluation time: 37.47 seconds
- Throughput: 133.44 images/second
- Serialized state-dict size: 44,960,715 bytes (44.96 MB)

### Interpretation

Validation accuracy decreased from 39.00% for the extended baseline to 38.84% after pruning, a reduction of 0.16 percentage points. The validation loss changed from 2.3520 to 2.3509.

The serialized state-dict size remained approximately 44.96 MB. This is expected for unstructured pruning when zero-valued weights are stored in dense tensors.

### Limitations

The evaluation used a single validation run. Timing measurements are preliminary and should not be interpreted as evidence of a reliable speedup. The held-out test set remains reserved for final evaluation.

## Extended Baseline — 40% Global Magnitude Pruning

### Objective

Evaluate the effect of applying 40% global unstructured magnitude pruning to the extended ResNet-18 baseline.

### Method

The extended baseline checkpoint, `models/resnet18_baseline_extended.pth`, was loaded. Global L1 unstructured pruning was applied to the convolutional weights and final fully connected layer weights. Pruning was then made permanent by removing the pruning reparameterization.

The pruned model was evaluated on the 5,000-image validation subset and saved as `models/resnet18_extended_pruned_40.pth`.

### Results

- Target pruning ratio: 40%
- Selected weights: 11,210,432
- Zero weights after pruning: 4,484,173
- Measured sparsity: 40.00%
- Validation loss: 2.3791
- Validation accuracy: 38.24%
- Validation Macro-F1: 0.3743
- Evaluation time: 38.40 seconds
- Throughput: 130.22 images/second
- Serialized state-dict size: 44,960,715 bytes (44.96 MB)

### Interpretation

Validation accuracy decreased from 39.00% for the extended baseline to 38.24% after 40% pruning, a reduction of 0.76 percentage points. The 40% pruning result also had lower validation accuracy than the 20% pruning result of 38.84%.

The serialized state-dict size remained approximately 44.96 MB. This is expected for unstructured pruning when zero-valued weights are stored in dense tensors.

### Limitations

The evaluation used a single validation run. Timing measurements are preliminary and should not be interpreted as evidence of a reliable speedup. The model was not fine-tuned after pruning, and the held-out test set remains reserved for final evaluation.

## Extended Baseline — Static INT8 Quantization

### Objective

Evaluate static INT8 quantization on the extended ResNet-18 baseline to measure its effect on validation performance and serialized model size.

### Method

The extended baseline checkpoint, `models/resnet18_baseline_extended.pth`, was loaded. FX graph-mode static quantization was prepared using the oneDNN backend and calibrated with 1,024 training images. The converted model was evaluated on the 5,000-image validation subset.

The quantized state dictionary was saved as `models/resnet18_extended_static_int8.pth`.

### Results

- Quantization backend: oneDNN
- Calibration images: 1,024
- Validation images: 5,000
- Validation loss: 2.3513
- Validation accuracy: 38.88%
- Validation Macro-F1: 0.3762
- Evaluation time: 12.21 seconds
- Throughput: 409.62 images/second
- Extended FP32 baseline size: 44,960,587 bytes
- Quantized state-dict size: 11,355,075 bytes (11.36 MB)
- Measured state-dict size reduction: 74.74%

### Interpretation

Validation accuracy decreased from 39.00% for the extended FP32 baseline to 38.88% after static INT8 quantization, a reduction of 0.12 percentage points. Validation loss remained close to the baseline value.

The serialized state-dict size decreased by 74.74%, from approximately 44.96 MB to 11.36 MB. This demonstrates substantial storage reduction with little observed change in validation accuracy.

### Limitations

The reported file size refers to the serialized state dictionary, not necessarily a complete standalone deployment package. Evaluation timing comes from a single run and should be treated as preliminary. A controlled repeated benchmark is needed before making reliable speed comparisons.

The current FX quantization APIs emitted deprecation warnings in the installed PyTorch environment. The experiment completed successfully, but future compatibility may require migration to newer quantization APIs.

The held-out test set remains reserved for final evaluation.

## Extended Baseline — 20% Pruning + Static INT8 Quantization

### Objective

Evaluate whether applying 20% global magnitude pruning before static INT8 quantization provides benefits beyond static INT8 quantization alone.

### Method

The extended ResNet-18 FP32 checkpoint was loaded. Global unstructured L1 magnitude pruning was applied to convolutional weights and the final fully connected layer, with a target pruning amount of 20%. Pruning was made permanent before FX graph-mode static INT8 quantization.

Quantization used the oneDNN backend and calibration on 1,024 images from the training subset. Evaluation used the project's fixed 5,000-image validation subset. The official 10,000-image CIFAR-100 test set was not used in this corrected run.

### Results

| Metric              | Extended FP32 Baseline | 20% Pruning + Static INT8 |
| ------------------- | ---------------------: | ------------------------: |
| Validation images   |                  5,000 |                     5,000 |
| Validation loss     |                 2.3520 |                    2.3498 |
| Validation accuracy |                 39.00% |                    38.80% |
| Validation Macro-F1 |                 0.3767 |                    0.3768 |
| Evaluation time     |          38.67 seconds |              9.64 seconds |
| Throughput          |   129.29 images/second |      518.57 images/second |
| Checkpoint size     |       44,960,587 bytes |          11,356,617 bytes |

- Measured pruning sparsity before quantization: 20.00%
- Calibration images: 1,024
- Checkpoint size reduction relative to the extended FP32 baseline: 74.74%
- Accuracy change relative to the extended FP32 baseline: -0.20 percentage points

### Comparison with Static INT8 Alone

Static INT8 alone achieved 38.88% validation accuracy, a Macro-F1 of 0.3762, a checkpoint size of 11,355,075 bytes, and measured throughput of 409.62 images/second in its earlier run.

The combined model achieved 38.80% validation accuracy and a Macro-F1 of 0.3768. Its checkpoint was 1,542 bytes larger than the static INT8-only checkpoint. The combined run showed higher throughput, but this difference is preliminary because the timing results come from single runs and may be affected by measurement variability.

### Interpretation

Static INT8 quantization accounts for most of the observed checkpoint size reduction. Applying 20% magnitude pruning before quantization did not provide additional meaningful storage savings and produced a small decrease in validation accuracy compared with static INT8 alone.

Although the combined run recorded higher throughput, repeated controlled benchmarking is required before concluding that pruning improves inference speed.

### Limitations

Pruning produced zeros in dense model tensors. This does not guarantee sparse storage or sparse-kernel acceleration after quantization. The reported checkpoint size refers to the serialized state dictionary, not a complete deployment package.

The results are based on the fixed validation subset. The official test set remains reserved for final evaluation. The timing measurements are preliminary and should be repeated under controlled conditions.

The installed FX quantization APIs emitted deprecation warnings, but the experiment completed successfully.

## Repeated CPU Inference Benchmark

### Objective

Compare CPU inference performance of the extended FP32 baseline, static INT8 model, and combined 20% pruning + static INT8 model.

### Method

The models were benchmarked on CPU using two batch sizes: 1 and 32. Each benchmark included 10 warm-up runs and 5 measured repetitions. The median latency and throughput were reported for each configuration.

The benchmark used 2 CPU threads and the oneDNN quantization backend. Static INT8 models were reconstructed using FX graph-mode quantization and calibrated with 1,024 images from the training subset.

Each measured repetition processed up to 100 batches from the validation DataLoader. Therefore, the reported timing includes data-loading overhead and should not be interpreted as pure model-only inference latency.

### Results — Latest Benchmark Session

| Model                     | Batch Size | Median Latency (ms/image) | Median Throughput (images/second) |
| ------------------------- | ---------: | ------------------------: | --------------------------------: |
| Extended Baseline FP32    |          1 |                    17.432 |                             57.37 |
| Static INT8               |          1 |                    11.325 |                             88.30 |
| 20% Pruning + Static INT8 |          1 |                    11.554 |                             86.55 |
| Extended Baseline FP32    |         32 |                     9.802 |                            102.02 |
| Static INT8               |         32 |                     2.284 |                            437.81 |
| 20% Pruning + Static INT8 |         32 |                     2.293 |                            436.19 |

### Interpretation

In the latest benchmark session, static INT8 quantization achieved higher throughput than the FP32 baseline at both batch sizes.

At batch size 1, static INT8 throughput was approximately 1.54 times the FP32 throughput. At batch size 32, it was approximately 4.29 times the FP32 throughput.

The combined pruning + INT8 model did not show a consistent speed advantage over static INT8 alone. In the latest session, its throughput was slightly lower than static INT8 at both batch sizes.

### Limitations

The results are preliminary and varied between benchmark sessions. CPU load, thermal conditions, and data-loading overhead may affect the measurements. More controlled model-only timing would be needed for a precise inference-latency comparison.

The benchmark reports performance on the validation pipeline, not final test-set performance. The official CIFAR-100 test set remains reserved for final evaluation.

The installed FX quantization APIs emitted deprecation warnings. These warnings did not stop the benchmark from completing.

## Final Evaluation on the Official CIFAR-100 Test Set

### Experimental Setup

The three selected checkpoints were evaluated on the official CIFAR-100 test set containing 10,000 images. The test set was kept separate from training and quantization calibration. Experiments were performed on CPU using two PyTorch threads.

### Results

| Model                     | Test Accuracy (%) | Macro-F1 | Checkpoint Size (MB) | Throughput (images/s) |
| ------------------------- | ----------------: | -------: | -------------------: | --------------------: |
| Extended FP32 baseline    |             38.94 |   0.3788 |                44.96 |                125.08 |
| Static INT8               |             38.66 |   0.3759 |                11.36 |                465.29 |
| 20% Pruning + Static INT8 |             38.71 |   0.3765 |                11.36 |                470.11 |

### Analysis

Static INT8 quantization reduced the serialized checkpoint size by approximately 74.7% compared with the extended FP32 baseline. Test accuracy decreased by 0.28 percentage points, while measured throughput increased by approximately 3.72 times.

The combined 20% pruning and INT8 model achieved 38.71% test accuracy and 470.11 images per second. Compared with INT8 alone, the combined model was only marginally faster in this run and had a slightly larger checkpoint. These results do not establish a meaningful practical advantage for adding unstructured pruning to INT8 quantization.

### Limitations

Throughput measurements depend on the hardware, software environment, batch size, and measurement procedure. The observed difference between the two INT8 models is small and should not be interpreted as a reliable speed advantage without repeated controlled benchmarks.

## Class-Level Error Analysis

### Overall Results

The extended FP32 baseline achieved 38.94% accuracy on the official CIFAR-100 test set.

Per-class accuracy was calculated for all 100 classes. The lowest accuracies included otter (2%), squirrel (3%), turtle (4%), crab (9%), and raccoon (10%).

### Common Misclassifications

The confusion matrix revealed several frequent errors:

- Otter was predicted as seal and beaver 12 times each.
- Squirrel was predicted as porcupine 9 times.
- Turtle was predicted as shark, ray, and dolphin 7 times each.
- Crab was predicted as lobster 7 times.
- Raccoon was predicted as wolf 9 times.

### Interpretation

These results show that the baseline model performs unevenly across CIFAR-100 classes. Some animal categories are particularly difficult for the model to distinguish. The confusion matrix helps identify specific class pairs associated with errors.

These observations describe model behavior; they do not establish the underlying cause of the errors.

### Saved Analysis Artifacts

- `reports/class_error_analysis.csv`: per-class accuracy for all 100 classes.
- `figures/class_accuracy_worst15.png`: visualization of the 15 classes with the lowest accuracy.
- `reports/confusion_matrix.csv`: full confusion matrix.
- `figures/confusion_matrix.png`: confusion matrix visualization.

## Reproducibility and Software Environment

### Software Environment

- Operating system: Windows 64-bit
- Python: 3.11.9
- PyTorch: 2.14.1+cpu
- Torchvision: 0.29.1+cpu
- Scikit-learn: 1.9.1
- Quantization backend: oneDNN (`onednn`)

### Runtime Configuration

- Final test evaluation: CPU with 2 PyTorch threads.
- Calibration images: 1,024 training-subset images per quantized model.
- Static INT8 calibration: first 1,024 images of `training_subset`.
- Combined pruning and INT8 calibration: 1,024 indices selected using seed 42.
- Test set: 10,000 official CIFAR-100 test images.

### Reproducibility Notes

Reproducing the results requires the same model architecture, checkpoint files, dataset split, preprocessing, calibration procedure, quantization backend, and compatible software versions. CPU timing may vary with hardware load and runtime conditions.

## Scientific Interpretation of Compression Results

### Effectiveness of Static INT8 Quantization

On the official CIFAR-100 test set, static INT8 quantization reduced the serialized checkpoint size from 44,960,587 bytes to 11,355,075 bytes, a reduction of approximately 74.74%. Test accuracy decreased from 38.94% to 38.66%, a reduction of 0.28 percentage points. Measured throughput increased from 125.08 to 465.29 images per second on the tested CPU setup.

These results indicate a favorable storage-accuracy trade-off under the evaluated configuration. Throughput results should be interpreted in the context of the measurement procedure and hardware.

### Additional Effect of Unstructured Pruning

The combined 20% pruning and static INT8 model achieved 38.71% test accuracy, compared with 38.66% for static INT8 alone. The difference was 0.05 percentage points. Throughput increased from 465.29 to 470.11 images per second, while the serialized checkpoint became 1,542 bytes larger.

The observed differences are small and do not establish a meaningful practical advantage from adding unstructured pruning to static INT8 quantization in this experiment.

### Limitations and Scope of Conclusions

Unstructured pruning creates zero-valued weights but does not necessarily reduce serialized model size or inference time unless the storage format and execution method exploit sparsity.

The reported checkpoint sizes refer to the saved state dictionaries, not necessarily the complete storage or memory requirements of a deployed application. Throughput measurements are specific to the tested CPU environment and measurement procedure.

The current results support conclusions about the evaluated models and configuration. Further controlled repeated measurements would be needed to establish the reliability of small throughput differences or generalize the findings to other hardware.
