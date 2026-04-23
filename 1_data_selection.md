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

# Step 1: Data Selection 


## 1. Read data file and choose the IDs with the main disease for the current project

```python
from paths import PM
from helpers.constants import ID_COL
```

```python
input_path = PM.raw_targets_file
temp_output_path = PM.processed_targets_file

keywords: list[str] = ["妊娠合并甲状腺功能减退"] # Target disease of the study
```

### Data Structure:

#### Medical Outcomes:

1. Other Outcome (implied if all others are negative)
2. 胎膜早破
3. 胎儿宫内窘迫
4. 巨大儿
5. 子痫前期
6. 分娩时I度会阴裂伤
7. 分娩时II度会阴裂伤
8. 羊水污染I度
9. 羊水污染II度
10. 羊水污染III度

#### Medical Conditions:

1. Other Condition (implied if all others are negative)
2. 妊娠期糖尿病
3. 妊娠期高血压
4. 妊娠合并肝损害
5. 妊娠合并肝内胆汁淤积症

<br>
<table>
    <tbody>
        <tr>
            <td>匹配ID</td>
            <td>胎膜早破</td>
            <td>胎儿宫内窘迫</td>
            <td>巨大儿</td>
            <td>子痫前期</td>
            <td>分娩时会阴裂伤</td>
            <td>羊水污染</td>
            <td>妊娠期糖尿病</td>
            <td>妊娠期高血压</td>
            <td>妊娠合并肝损害</td>
            <td>妊娠合并肝内胆汁淤积症</td>
        </tr>
        <tr>
            <td>(ID)</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
            <td>0 or 1</td>
        </tr>
    </tbody>
</table>
<br>

```python
condition1 = "妊娠期糖尿病"
condition2 = "妊娠期高血压"
condition3 = "妊娠合并肝损害"
condition4 = "妊娠合并肝内胆汁淤积症"

outcome1 = "胎膜早破"
outcome2 = "胎儿宫内窘迫"
outcome3 = "巨大儿"
outcome4 = "子痫前期"

outcome5 = "分娩时会阴裂伤"
outcome5_1 = "分娩时I度会阴裂伤"
outcome5_2 = "分娩时II度会阴裂伤"

outcome6 = "羊水污染"
outcome6_1 = "羊水污染I度"
outcome6_2 = "羊水污染II度"
outcome6_3 = "羊水污染III度"
```

```python
headers = [ID_COL, 
           outcome1, outcome2, outcome3, outcome4, outcome5, outcome6,
           condition1, condition2, condition3, condition4]
```

```python
# Write headers to the output file
with open(temp_output_path, "w", encoding="utf-8") as f:
    f.write(",".join(headers) + "\n")

# Populate the output file with binary data
with open(input_path) as file:
    for line in file:
        # Initialize values
        values: str = ""
                
        line = line.strip()
        segments = line.split(",")
        
        # Check keyword(s) in each segment
        for keyword in keywords:
            if keyword in segments:
                # Save the ID of the line
                values += segments[0].strip() + ","
                
                # Check outcomes
                if outcome1 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if outcome2 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if outcome3 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if outcome4 in segments:
                    values += "1,"
                else:
                    values += "0,"
                    
                # Multi-degree outcomes
                if outcome5_2 in segments:
                    values += "1,"
                elif outcome5_1 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if outcome6_3 in segments:
                    values += "1,"
                elif outcome6_2 in segments:
                    values += "1,"
                elif outcome6_1 in segments:
                    values += "1,"
                else:
                    values += "0,"
                
                # Check conditions
                if condition1 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if condition2 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if condition3 in segments:
                    values += "1,"
                else:
                    values += "0,"
                if condition4 in segments:
                    values += "1"
                else:
                    values += "0"
                    
                # Append the values to the output file
                with open(temp_output_path, "a", encoding="utf-8") as f:
                    f.write(values + "\n")
```

## 2. Extract data from the main data source

```python
import polars as pl
```

```python
# Get column names to use
with open(PM.feature_columns_file, "r", encoding="utf-8-sig") as f:
    columns = f.readline().strip().split(",")
print(columns)
```

```python
# Read files to concatenate (read all data as strings)
df_base = pl.read_csv(temp_output_path, infer_schema=False)
df_raw = pl.read_csv(PM.raw_data_file, columns=columns, infer_schema=False)
```

## 3. Drop Targets with no positive cases

```python
target_columns = [outcome1, outcome2, outcome3, outcome4, outcome5, outcome6]
condition_columns = [condition1, condition2, condition3, condition4]

_df_targets = df_base.select([pl.col(col).cast(pl.Int32) for col in target_columns])
df_targets_clean = _df_targets.select([col for col in _df_targets.columns if _df_targets[col].sum() > 0])

print(_df_targets.shape)
print(_df_targets.columns)
print(df_targets_clean.shape)
print(df_targets_clean.columns)
```

```python
df_base_v2 = df_base.select([ID_COL] + df_targets_clean.columns + condition_columns)
print(df_base.shape, df_base_v2.shape)
```

## 4. Merge Dataframe 

```python
# Perform a left join
df_final = df_base_v2.join(df_raw, on=ID_COL, how="left")
```

```python
# Save final data
df_final.write_csv(PM.clean_data_cn_file)
```
