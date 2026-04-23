---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: pregnancy-hypothyroidism
    language: python
    name: python3
---

```python
from ml_tools.data_exploration import check_class_balance, summarize_dataframe, drop_constant_columns
from ml_tools.resampling import DragonResampler
from ml_tools.utilities import save_dataframe_with_schema, load_dataframe_greedy, distribute_dataset_by_target
from ml_tools.path_manager import sanitize_filename
from ml_tools.schema import FeatureSchema

from paths import PM
from helpers.constants import TARGETS
```

## 1. Load data

```python
df = load_dataframe_greedy(PM.mice_datasets)
```

```python
df = drop_constant_columns(df)
```

```python
feature_schema = FeatureSchema.from_json(PM.engineering_artifacts)
```

```python
summarize_dataframe(df)
```

## 2. Check original class balance

```python
check_class_balance(df=df,
                    target=TARGETS,
                    plot_to_dir=PM.resampling,
                    plot_filename="Class_Balance_Original")
```

## 3. Distribute training datasets

```python
dataset_iterator = distribute_dataset_by_target(df_or_path=df, target_columns=TARGETS)
```

## 4. Resample and save train datasets

```python
for target_name, df_split in dataset_iterator:
    sampler = DragonResampler(target_column=target_name, return_pandas=True)
    
    df_balanced = sampler.balance_classes(df=df_split)
    
    _ = check_class_balance(df=df_balanced,
                            target=target_name,
                            plot_to_dir=PM.resampling,
                            plot_filename=f"Class_Balance_{target_name}")
    
    csv_filename = sanitize_filename(target_name) + '.csv'
    
    save_dataframe_with_schema(df=df_balanced, full_path=PM.train_datasets / csv_filename, schema=feature_schema)
```
