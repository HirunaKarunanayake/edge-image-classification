# Dataset Setup

This project uses the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle.
The images are not stored in this repository because of their size.

## Steps

1. Download the dataset from Kaggle (a free account is required).
2. Extract it into a `data/` folder in the project root.
3. Make sure the structure matches this exactly:

```
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
        └── (same six class folders)
```

## Verify the setup

Run from the project root:

```
python src/check_data.py
python src/check_sizes.py
```

## Common problems

- **Folder not found:** the Kaggle zip often creates an extra outer folder. Move `seg_train` and `seg_test` directly into `data/`.
- **Wrong working directory:** always run scripts from the project root, not from inside `src/`.
