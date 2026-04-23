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

# Step 4: Feature Engineering

```python
from ml_tools.data_exploration import info
info()
```

```python
from ml_tools.data_exploration import (summarize_dataframe,
                                       drop_macro,
                                       clean_column_names,
                                       plot_value_distributions,
                                       split_features_targets,
                                       plot_correlation_heatmap,
                                       finalize_feature_schema)
from ml_tools.utilities import load_dataframe, save_dataframe_with_schema, merge_dataframes
from ml_tools.plot_fonts import configure_cjk_fonts

from paths import PM 
from helpers.constants import TARGETS

configure_cjk_fonts()
```

## 1. Load dataframe

```python
df, _ = load_dataframe(df_path=PM.processed_data_file, kind="pandas")
```

```python
summarize_dataframe(df)
```

## 2. Drop features and samples if necessary

```python
df_clean_I = drop_macro(df=df,
                      log_directory=PM.engineering_plots,
                      targets=TARGETS,
                      skip_targets=True,
                      threshold=0.7)
```

```python
df_clean_II = df_clean_I.copy()
df_clean_II = clean_column_names(df=df_clean_II, replacement_char='(', replacement_pattern='（')
df_clean_II = clean_column_names(df=df_clean_II, replacement_char=')', replacement_pattern='）')
df_clean_II = clean_column_names(df=df_clean_II, replacement_char='gamma', replacement_pattern='γ')
```

```python
df_clean = df_clean_II
```

```python
summarize_dataframe(df_clean)
```

## 3. Value Distribution

```python
plot_value_distributions(df=df_clean, save_dir=PM.engineering_plots,)
```

## 4. Dataset splits

```python
df_features, df_targets = split_features_targets(df=df_clean, targets=TARGETS)
```

## 5. Plots

```python
plot_correlation_heatmap(df=df_features, plot_title="Features", save_dir=PM.engineering_plots)
```

## 6. Make Schema

```python
feature_schema = finalize_feature_schema(df_features=df_features, categorical_mappings=None)
```

## 7. Save Artifacts

```python
feature_schema.to_json(PM.engineering_artifacts)
feature_schema.save_artifacts(PM.engineering_artifacts)
```

## 8. Save DataFrame

```python
df_final = merge_dataframes(df_features, df_targets)
```

```python
summarize_dataframe(df_final)
```

```python
save_dataframe_with_schema(df=df_final, full_path=PM.engineering_data_file, schema=feature_schema)
```
