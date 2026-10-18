## Model A vs Model B (Adam)

| Model | Total params | Trainable params | MACs / image | Size (KB, float32) | Size (KB, .keras file) | Epoch time (s) | Test acc | Macro precision | Macro recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 250,886 | 250,182 | 50,733,440 | 980.0 | 3033.1 | 111.4 | 0.8716 | 0.8726 | 0.8735 |
| B | 38,817 | 38,113 | 6,271,360 | 151.6 | 549.7 | 133.3 | 0.8224 | 0.8252 | 0.8250 |

## Optimizer comparison

| Run | Best epoch | Best val loss | Best val acc | Test acc |
| --- | --- | --- | --- | --- |
| A_adam | 18 | 0.3836 | 0.8644 | 0.8716 |
| A_sgd_momentum | 18 | 0.4216 | 0.8538 | 0.8427 |
| A_sgd | 13 | 0.6131 | 0.7719 | 0.7693 |
| B_adam | 20 | 0.4773 | 0.8245 | 0.8224 |
| B_sgd_momentum | 18 | 0.5340 | 0.8013 | 0.7947 |
| B_sgd | 20 | 0.5912 | 0.7872 | 0.7724 |

## Model B vs fine-tuned lightweight SOTA

| Model | Total params | Size (MB) | Epoch time (s) | Test acc | Macro precision | Macro recall |
| --- | --- | --- | --- | --- | --- | --- |
| Model B | 38,817 | 0.148 | 133.3 | 0.8224 | 0.8252 | 0.8250 |
| efficientnetb0 | 4,057,257 | 15.48 | 576.7 | 0.8911 | 0.8924 | 0.8933 |
| mobilenetv2 | 2,265,670 | 8.64 | 460.0 | 0.8759 | 0.8792 | 0.8778 |
