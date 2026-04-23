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

# Step 2: Data Preprocessing


## 1. Load data

```python
from paths import PM
import polars as pl
from helpers.constants import ID_COL
```

```python
# Revised version on 2025-02-21, 2025-03-20, and 2026-01-04
clean_data_path = PM.clean_data_cn_file
```

```python
columns_to_fix = {
    "38-40周血葡萄糖": pl.Utf8,
    "30.1-37.6周血葡萄糖": pl.Utf8,
    "10.1-20周促甲状腺激素受体抗体": pl.Utf8,
    "21-40周促甲状腺激素受体抗体": pl.Utf8,
    "25.1-32w_脐动脉S/D": pl.Utf8,
    "18.1-25w_脐动脉RI": pl.Utf8,
    "1-10周促甲状腺激素受体抗体": pl.Utf8,
    "1-11.6周糖化血红蛋白": pl.Utf8,
    "25.1-33.6周γ-谷氨酰转肽酶": pl.Utf8,
    "22.1-32周乳酸脱氢酶": pl.Utf8,
    "25.1-33.6周谷丙转氨酶": pl.Utf8,
    "25.1-33.6周谷草转氨酶": pl.Utf8,
    "1-10周TGAB": pl.Utf8,
    "25.1-32w_羊水深度": pl.Utf8,
    "1-12.6周谷丙转氨酶": pl.Utf8,
    "18.1-25w_双顶径": pl.Utf8,
    "13-25周γ-谷氨酰转肽酶": pl.Utf8,
    "1-22周乳酸脱氢酶": pl.Utf8,
    "13-25周谷丙转氨酶": pl.Utf8,
    "13-25周谷草转氨酶": pl.Utf8,
    "38-40周谷丙转氨酶": pl.Utf8,
    "32.1-40周乳酸脱氢酶": pl.Utf8,
} # Obtained by trial and error

df = pl.read_csv(clean_data_path, infer_schema=True, schema_overrides=columns_to_fix)
df.shape
```

```python
# Check for manually identified columns
for col in columns_to_fix.keys():
    if col not in df.columns:
        raise ValueError(f"Column to fix '{col}' not found in the dataframe.")
```

```python
# Split dataframe into two dataframes: one with columns to fix and the rest
df_ok = df.drop(list(columns_to_fix.keys()))

df_to_fix = df.select([ID_COL] + list(columns_to_fix.keys()))

print(df_ok.shape, df_to_fix.shape)
```

## 2. Delete features with more empty values than a given threshold value

```python
def delete_columns_with_nulls(df: pl.DataFrame, threshold_percentage: float=0.5):
    threshold = threshold_percentage * df.height

    # Get null counts for all columns in a single operation
    null_counts = df.null_count()

    # Create a Series mapping column names to their null counts
    null_counts_dict = dict(zip(null_counts.columns, null_counts.row(0)))

    # Identify columns to drop
    cols_to_drop = [col for col, null_count in null_counts_dict.items() if null_count >= threshold]

    # Drop columns iand return the new DataFrame
    return df.drop(cols_to_drop)
```

```python
df_ok_2 = delete_columns_with_nulls(df_ok)
df_to_fix_2 = delete_columns_with_nulls(df_to_fix)
print(df_ok_2.shape, df_to_fix_2.shape)
```

## 3. Cast and amend data entries

```python
def _cast_value(value, column_name, row_idx):
    '''
    Try to cast a value to an integer or float. If it fails, prompt the user to enter a corrected value or set it as null.
    '''
    while True:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                user_input = input(
                    f"Error casting value '{value}' in column '{column_name}', row {row_idx}. "
                    f"Enter a corrected value or type 'None'/'Null' to set it as null: "
                ).strip()
                
                if user_input.lower() in ["none", "null"]:
                    return None
                else:
                    value = user_input  # Update the value and retry casting
                    

def fix_values_dtype(df: pl.DataFrame):
    # Fix columns
    processed_columns = dict()

    for column in df.columns:
        if column == ID_COL:
            processed_columns[column] = df[column].to_list()
            continue
        
        processed_values = []
        for row_idx, value in enumerate(df[column].to_list(), start=1):
            if value is None:
                processed_values.append(None)
                continue
            processed_value = _cast_value(value, column, row_idx)
            processed_values.append(processed_value)
        
        # Check if conversion to all floats is needed (avoid integers and float mixture)
        has_floats = any(isinstance(v, float) for v in processed_values)
        if has_floats:
            processed_values = [
                float(v) if v is not None else None 
                for v in processed_values
            ]
        
        processed_columns[column] = processed_values

    # Create a new DataFrame with the processed data
    return pl.DataFrame(processed_columns)
```

```python
df_to_fix_3 = fix_values_dtype(df_to_fix_2)
print(df_to_fix_3.shape)
```

```python
# Validate row counts
print(df_ok_2.shape, df_to_fix_3.shape)
assert df_ok_2.height == df_to_fix_3.height, "Row counts do not match!"
```

```python
# Join the two DataFrames on ID
merged_df = df_ok_2.join(df_to_fix_3, on=ID_COL, how="inner")
print(merged_df.shape)
```

## 3.1 Fix Faulty Entries

```python
# Fix faulty entries in the "身高" column. If the data value is less than 150, set it to empty. 
merged_df = merged_df.with_columns(
    pl.when(pl.col("身高") < 150.0)
    .then(None)
    .otherwise(pl.col("身高"))
    .alias("身高")
)
print(merged_df.shape)
```

## 4. Audit translation for column names

```python
from ml_tools.utilities import audit_column_translation, save_dataframe
```

```python
audit_column_translation(df_or_path=merged_df, mapper=PM.translation_file)
```

## 5. Save processed data

```python
save_dataframe(df=merged_df, full_path=PM.processed_data_cn_file)
```
