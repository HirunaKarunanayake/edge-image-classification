
# Edge Image Classification

**EN3150 – Assignment 03 | Edge Image Classification**

This project develops and evaluates convolutional neural networks (CNNs) for classifying natural scene images on resource-constrained edge devices.

The main objective is to investigate the trade-off between classification accuracy, model complexity, computational requirements, and storage size.

Four models are evaluated:

1. **Model A:** A conventional CNN used as the baseline.
2. **Model B:** A lightweight CNN using depthwise separable convolutions.
3. **MobileNetV2:** An ImageNet-pretrained model adapted through transfer learning.
4. **EfficientNetB0:** An ImageNet-pretrained model adapted through transfer learning.

The project also compares three optimization algorithms: Adam, SGD, and SGD with momentum.

All experiments were conducted using TensorFlow/Keras on a CPU-based system.

---

## 1. Dataset

The project uses the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

The dataset contains natural scene images belonging to six classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The original training and testing images were combined and redistributed using a stratified split with a fixed random seed of 42.

### Dataset distribution

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 11,920 | 70% |
| Validation | 2,552 | 15% |
| Testing | 2,562 | 15% |
| **Total** | **17,034** | **100%** |

All images are resized to **64 × 64 pixels**, converted to RGB, and normalized to the range [0, 1].

The dataset split is recorded in `results/split.csv` to ensure that all models use identical training, validation, and testing samples.

### Sample images

![Dataset samples](plots/samples.png)

### Downloading the dataset

The original image dataset is not included in this repository because of its size.

Download it from Kaggle and extract the folders into the following structure:

```text
data/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_test/
    └── seg_test/
        ├── buildings/
        ├── forest/
        ├── glacier/
        ├── mountain/
        ├── sea/
        └── street/
```

The dataset must be placed correctly before running the preprocessing, training, or evaluation scripts.

---

## 2. Project Structure

```text
edge-image-classification/
│
├── data/                  # Dataset (raw images excluded from Git)
│
├── models/                # Trained model checkpoints
│
├── plots/                 # Training curves and confusion matrices
│
├── results/               # Metrics, histories, and dataset split
│
├── src/
│   ├── check_data.py
│   ├── check_sizes.py
│   ├── view_samples.py
│   ├── split_data.py
│   ├── data.py
│   ├── modelA.py
│   ├── modelB.py
│   ├── train.py
│   ├── sota.py
│   ├── evaluate.py
│   ├── plot_utils.py
│   └── report_plots.py
│
├── .gitignore
├── .gitattributes
└── README.md
```

### Main directories

- `src/`: Python scripts for preprocessing, model development, training, and evaluation.
- `data/`: Original Intel dataset images.
- `models/`: Saved Keras model checkpoints.
- `results/`: Dataset splits, training histories, and evaluation metrics.
- `plots/`: Generated figures, training curves, and confusion matrices.

---

## 3. Model Architectures

### 3.1 Model A – Standard CNN

Model A is the baseline CNN consisting of four convolutional blocks.

Each block contains:

- A 3 × 3 convolution
- Batch normalization
- ReLU activation
- 2 × 2 max pooling

The convolutional blocks use 32, 64, 128, and 128 filters respectively.

The feature extraction stage is followed by global average pooling, dropout (0.3), a dense layer with 64 neurons, and a six-class softmax output layer.

| Property | Value |
|---|---:|
| Total parameters | 250,886 |
| Trainable parameters | 250,182 |
| MACs per image | 50,733,440 |
| Float32 weight storage | 980.02 KiB |
| Saved model size | 3,033.08 KiB |

Model A provides a baseline for evaluating the accuracy and efficiency of the lightweight architecture.

### 3.2 Model B – Lightweight CNN

Model B replaces the conventional convolutional layers with depthwise separable convolutions.

A depthwise separable convolution performs spatial filtering independently for each input channel, followed by a 1 × 1 pointwise convolution that combines the channels.

This reduces the number of parameters and multiply-accumulate operations.

The remaining architecture is similar to Model A, with batch normalization, ReLU, max pooling, global average pooling, dropout (0.2), and a six-class classifier.

| Property | Value |
|---|---:|
| Total parameters | 38,817 |
| Trainable parameters | 38,113 |
| MACs per image | 6,271,360 |
| Float32 weight storage | 151.63 KiB |
| Saved model size | 549.69 KiB |

Model B satisfies the assignment constraint of fewer than 100,000 trainable parameters.

Compared with Model A, Model B uses approximately **84.5% fewer parameters** and **87.6% fewer MACs**.

### 3.3 Transfer Learning Models

Two ImageNet-pretrained architectures are evaluated:

**MobileNetV2**

- Lightweight architecture based on inverted residual blocks.
- Pretrained ImageNet feature extractor.
- Global average pooling and a new six-class classification head.

**EfficientNetB0**

- Architecture designed for efficient scaling of network depth, width, and resolution.
- Pretrained ImageNet feature extractor.
- Global average pooling and a new six-class classification head.

Both models use a two-stage transfer learning procedure.

---

## 4. Training Methodology

### 4.1 Data Augmentation

During training, random transformations are applied to improve generalization:

- Horizontal flipping
- Random rotation
- Random zooming

Data augmentation is disabled during validation and testing.

### 4.2 Training Configuration

The custom models use:

- Categorical cross-entropy loss
- Batch size of 32
- Maximum training duration configured per experiment
- Batch normalization momentum of 0.9
- Model checkpointing based on validation loss
- Learning rate reduction on validation-loss plateaus
- Early stopping when applicable

The optimizer configurations are:

| Model | Optimizer | Initial learning rate |
|---|---|---:|
| A | Adam | 0.001 |
| A | SGD | 0.005 |
| A | SGD + Momentum | 0.005 |
| B | Adam | 0.001 |
| B | SGD | 0.010 |
| B | SGD + Momentum | 0.010 |

SGD with momentum uses a momentum coefficient of 0.9.

The configurations are defined in `src/train.py`.

### 4.3 Transfer Learning

Both pretrained networks are trained in two stages.

**Stage 1 – Feature Extraction**

- Freeze the pretrained backbone.
- Train the newly added classification head.
- Learning rate: 0.001
- Duration: 5 epochs

**Stage 2 – Fine-Tuning**

- Unfreeze the backbone.
- Fine-tune the model using a lower learning rate.
- Learning rate: 0.0001
- Duration: 15 epochs

This allows the pretrained representations to adapt to the Intel scene classification task.

---

## 5. Installation and Setup

### Requirements

- Python 3.10–3.12
- TensorFlow
- NumPy
- Matplotlib
- Pillow
- scikit-learn

A CPU is sufficient to execute the project, although training can take a considerable amount of time.

### Step 1 – Clone the repository

```bash
git clone https://github.com/HirunaKarunanayake/edge-image-classification.git
cd edge-image-classification
```

Replace the placeholder with your GitHub repository URL.

### Step 2 – Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 – Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow numpy matplotlib pillow scikit-learn
```

If a verified `requirements.txt` is provided, the dependencies can instead be installed with:

```powershell
python -m pip install -r requirements.txt
```

### Step 4 – Download the dataset

Download the Intel Image Classification dataset from Kaggle and place the extracted images in the `data/` directory as described above.

Always execute the scripts from the project root directory.

---

## 6. Running the Project

### Step 1 – Inspect the dataset

```powershell
python src/check_data.py
python src/check_sizes.py
python src/view_samples.py
```

These scripts inspect the dataset and generate sample visualizations.

### Step 2 – Generate the dataset split

```powershell
python src/split_data.py
```

This generates `results/split.csv`.

If the split file already exists and corresponds to the dataset, reuse it to maintain consistency with the reported experiments.

### Step 3 – Verify the architectures

```powershell
python src/modelA.py
python src/modelB.py
python src/data.py
```

These scripts can be used to inspect model summaries and verify the input pipeline.

### Step 4 – Train the custom models

**Model A**

```powershell
python src/train.py A adam
python src/train.py A sgd
python src/train.py A sgd_momentum
```

**Model B**

```powershell
python src/train.py B adam
python src/train.py B sgd
python src/train.py B sgd_momentum
```

The training scripts save checkpoints, training histories, and learning curves.

### Step 5 – Train the pretrained models

```powershell
python src/sota.py mobilenetv2
python src/sota.py efficientnetb0
```

The first execution may require an internet connection to download pretrained ImageNet weights.

### Step 6 – Evaluate the models

```powershell
python src/evaluate.py
```

The evaluation process calculates test metrics and generates confusion matrices and comparison outputs.

---

## 7. Experimental Results

### 7.1 Final Test Performance

The following results were obtained using the held-out test dataset.

| Model | Test Accuracy | Macro Precision | Macro Recall |
|---|---:|---:|---:|
| Model A – Adam | 87.16% | 87.26% | 87.35% |
| Model B – Adam | 82.24% | 82.52% | 82.50% |
| MobileNetV2 | 87.59% | 87.92% | 87.78% |
| EfficientNetB0 | **89.11%** | **89.24%** | **89.33%** |

EfficientNetB0 achieves the highest test accuracy, followed by MobileNetV2 and Model A.

Model B achieves lower accuracy but offers a substantial reduction in computational and storage requirements.

### 7.2 Optimizer Comparison

The following values represent the validation performance at the epoch with the lowest validation loss.

| Model | Optimizer | Best Epoch | Validation Accuracy | Validation Loss |
|---|---|---:|---:|---:|
| A | Adam | 18 | 86.44% | 0.3836 |
| A | SGD | 13 | 77.19% | 0.6131 |
| A | SGD + Momentum | 18 | 85.62% | 0.4173 |
| B | Adam | 20 | 82.45% | 0.4773 |
| B | SGD | 20 | 78.72% | 0.5912 |
| B | SGD + Momentum | 18 | 80.13% | 0.5340 |

Adam achieved the best validation performance for both custom architectures and was therefore selected for the primary model comparison.

### 7.3 Resource Comparison

| Model | Total Parameters | Float32 Weight Storage | Saved Model Size |
|---|---:|---:|---:|
| Model A | 250,886 | 980.02 KiB | 3,033.08 KiB |
| Model B | 38,817 | 151.63 KiB | 549.69 KiB |
| MobileNetV2 | 2,265,670 | 8.64 MiB | 26.20 MiB |
| EfficientNetB0 | 4,057,257 | 15.48 MiB | 46.86 MiB |

Float32 weight storage is estimated from the parameter count, while saved model size represents the measured serialized checkpoint.

### 7.4 Computational Complexity

| Model | MACs per image |
|---|---:|
| Model A | 50.73 million |
| Model B | 6.27 million |

Model B requires approximately 8.1 times fewer MACs than Model A.

This makes it a more suitable candidate for deployment where computational resources are limited.

### 7.5 Training Time

The recorded mean training times were:

| Model | Mean Time per Epoch |
|---|---:|
| Model A | 111.41 s |
| Model B | 133.35 s |
| MobileNetV2 – Fine-tuning stage | 160.02 s |
| EfficientNetB0 – Fine-tuning stage | 576.71 s |

These measurements were collected from the experiment runs and should not be interpreted as controlled hardware benchmarks.

In particular, lower MAC counts do not always translate directly into faster CPU training because training performance also depends on implementation efficiency, memory operations, and software optimization.

---

## 8. Training Curves and Confusion Matrices

Training and validation accuracy/loss curves are saved in `plots/`.

These figures help identify:

- Convergence behavior
- Overfitting and underfitting
- Optimizer performance
- Learning rate effects
- Fine-tuning behavior

The pretrained models can exhibit temporary validation-loss fluctuations when the backbone is unfrozen at the beginning of fine-tuning.

Confusion matrices provide class-level information about correct classifications and common prediction errors.

The evaluation scripts generate the corresponding figures for the trained models.

---

## 9. Discussion

### Model A vs. Model B

Model A achieves higher classification accuracy than Model B.

However, Model B substantially reduces the number of trainable parameters, model storage requirements, and MAC operations.

This demonstrates the trade-off between predictive performance and computational efficiency.

### Transfer Learning

The pretrained MobileNetV2 and EfficientNetB0 models achieve competitive or superior classification accuracy by using representations learned from ImageNet.

EfficientNetB0 provides the best classification performance but requires substantially more storage and computational resources.

### Edge Deployment

For devices with strict memory and computational limits, Model B offers an attractive lightweight solution.

Model A provides a stronger accuracy baseline at a moderate increase in model complexity.

MobileNetV2 and EfficientNetB0 may be suitable for more capable edge platforms, depending on the available memory, inference runtime, and latency requirements.

Actual deployment suitability must be confirmed through device-specific inference benchmarking.

---

## 10. Reproducibility

The experiments use a fixed random seed of 42 and a common dataset split.

For consistent reproduction:

1. Use the same dataset and folder structure.
2. Reuse `results/split.csv`.
3. Use the same model configurations and learning rates.
4. Apply the same preprocessing and augmentation.
5. Evaluate all models using the same held-out test set.

Minor differences in results may occur because of software versions, floating-point behavior, CPU optimizations, and nondeterministic operations.

---

## 11. Git and Repository Management

The repository uses `.gitignore` to exclude unnecessary local files, including Python virtual environments, cache files, and raw dataset images.

Trained model checkpoints may be managed through Git Large File Storage (Git LFS).

The `.gitattributes` file defines the Git LFS rules for supported model file formats.

The source code, evaluation metrics, generated plots, and documentation are intended to remain accessible in the repository.

---

## 12. Conclusion

This project demonstrates the practical trade-offs involved in designing image classifiers for edge devices.

The standard CNN (Model A) achieved 87.16% test accuracy, while the lightweight depthwise separable CNN (Model B) achieved 82.24% with significantly fewer parameters and MAC operations.

Transfer learning improved classification performance, with EfficientNetB0 achieving the highest test accuracy of 89.11%.

The results show that model selection for edge deployment should consider not only classification accuracy but also memory requirements, computational complexity, and the capabilities of the target hardware.
