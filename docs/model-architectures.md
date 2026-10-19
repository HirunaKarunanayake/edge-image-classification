# Model Architectures

| Model | Type | Total Parameters | MACs per image | Saved Model Size |
|-------|------|------------------|----------------|------------------|
| Model A | Standard CNN | 250,886 | 50.73 M | 3,033.08 KiB |
| Model B | Depthwise separable CNN | 38,817 | 6.27 M | 549.69 KiB |
| MobileNetV2 | Pretrained (ImageNet) | 2,265,670 | not measured | 26.20 MiB |
| EfficientNetB0 | Pretrained (ImageNet) | 4,057,257 | not measured | 46.86 MiB |

## Model A
Four conv blocks (32, 64, 128, 128 filters), each with 3x3 conv, batch norm, ReLU and 2x2 max pooling,
followed by global average pooling, dropout (0.3), a 64-unit dense layer and a 6-class softmax.

## Model B
Same structure as Model A but using depthwise separable convolutions, with dropout 0.2.
About 84.5% fewer parameters and 87.6% fewer MACs than Model A.

## Transfer learning models
MobileNetV2 and EfficientNetB0 use a frozen pretrained backbone with a new 6-class head
(5 epochs, lr 0.001), then are fine-tuned (15 epochs, lr 0.0001).

## Test accuracy

| Model | Test Accuracy |
|-------|---------------|
| Model A (Adam) | 87.16% |
| Model B (Adam) | 82.24% |
| MobileNetV2 | 87.59% |
| EfficientNetB0 | 89.11% |