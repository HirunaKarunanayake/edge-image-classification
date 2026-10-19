# Usage Guide

Run all commands from the project root with your virtual environment active.

## 1. Check the dataset
```
python src/check_data.py
python src/view_samples.py
```

## 2. Create the split
```
python src/split_data.py
```

## 3. Train
```
python src/train.py A adam
python src/train.py B adam
python src/sota.py mobilenetv2
python src/sota.py efficientnetb0
```

## 4. Evaluate
```
python src/evaluate.py
```