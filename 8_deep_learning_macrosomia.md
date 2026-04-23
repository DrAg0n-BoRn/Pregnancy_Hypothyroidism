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
from ml_tools.ML_datasetmaster import DragonDataset
from ml_tools.ML_models import DragonNodeModel
from ml_tools.ML_configuration import (
    FormatBinaryClassificationMetrics,
    FinalizeBinaryClassification,
    DragonNodeParams
)

from ml_tools.ML_configuration import DragonTrainingConfig
from ml_tools.ML_trainer import DragonTrainer
from ml_tools.ML_callbacks import DragonModelCheckpoint, DragonPatienceEarlyStopping, DragonPlateauScheduler
from ml_tools.ML_utilities import build_optimizer_params, inspect_model_architecture
from ml_tools.utilities import load_dataframe_with_schema
from ml_tools.path_manager import sanitize_filename
from ml_tools.IO_tools import train_logger
from ml_tools.schema import FeatureSchema
from ml_tools.keys import TaskKeys
from torch.optim import AdamW

from paths import PM

# Choose Target
from helpers.constants import TARGET_MACROSOMIA as TARGET
```

```python
# Set paths
sanitized_target = sanitize_filename(TARGET)

PM.train_results = PM.results / sanitized_target
PM.train_artifacts = PM.train_results / "Artifacts"
PM.train_checkpoints = PM.train_results / "Checkpoints"
PM.train_evaluation = PM.train_results / "Evaluation"

PM.make_dirs()

# Local path constants
SCHEMA_PATH = PM.engineering_artifacts
TRAIN_DATASET_FILE = PM.train_datasets / (sanitized_target + '.csv')
TRAIN_ARTIFACTS_DIR = PM.train_artifacts
TRAIN_CHECKPOINTS_DIR = PM.train_checkpoints
TRAIN_EVALUATION_DIR = PM.train_evaluation
```

## 1. Config

```python
train_config = DragonTrainingConfig(
    validation_size=0.2,
    test_size=0.1,
    initial_learning_rate=0.0001,
    batch_size=24,
    task = TaskKeys.BINARY_CLASSIFICATION,
    device = "cuda:0",
    finalized_filename = f"node_{sanitized_target}",
    random_state=101,
    
    target=TARGET,
    early_stop_patience=20,
    scheduler_patience=3,
    scheduler_lr_factor=0.5,    
)
```

## 2. Load Schema and Dataframe

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)

df, _ = load_dataframe_with_schema(df_path=TRAIN_DATASET_FILE, schema=schema)
```

## 3. Make Datasets

```python
dataset = DragonDataset(pandas_df=df,
                        schema=schema,
                        kind=train_config.task,
                        feature_scaler="fit",
                        target_scaler="none",
                        validation_size=train_config.validation_size,
                        test_size=train_config.test_size,
                        random_state=train_config.random_state,
                        class_map={"Negative": 0, "Positive": 1})
```

## 4. Model and Trainer

```python
model_params = DragonNodeParams(
    schema=schema,
    out_targets=1,
    embedding_dim=32,
    num_trees=1024,
    num_layers=2,
    tree_depth=5,
    additional_tree_output_dim=3,
    input_dropout=0.0,
    embedding_dropout=0.0,
    choice_function='sparsemax',
    bin_function='sparsemoid',
    batch_norm_continuous=False
)

model = DragonNodeModel(**model_params)
# Initialize decision thresholds before training.
model.data_aware_initialization(train_dataset=dataset.train_dataset, num_samples=500)

# optimizer
optim_params = build_optimizer_params(model=model, weight_decay=0.001)
optimizer = AdamW(params=optim_params, lr=train_config.initial_learning_rate)

trainer = DragonTrainer(model=model,
                        train_dataset=dataset.train_dataset,
                        validation_dataset=dataset.validation_dataset,
                        kind=train_config.task,
                        optimizer=optimizer,
                        device=train_config.device,
                        checkpoint_callback=DragonModelCheckpoint(save_dir=TRAIN_CHECKPOINTS_DIR, 
                                                                  monitor="Validation Loss"),
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor="Validation Loss"),
                        lr_scheduler_callback=DragonPlateauScheduler(monitor="Validation Loss",
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),  
                        )
```

## 5. Training

```python
history = trainer.fit(save_dir=TRAIN_ARTIFACTS_DIR, epochs=500, batch_size=train_config.batch_size)
```

## 6. Evaluation

```python
trainer.evaluate(save_dir=TRAIN_EVALUATION_DIR,
                model_checkpoint="best",
                test_data=dataset.test_dataset,
                classification_threshold=0.5,
                val_format_configuration=FormatBinaryClassificationMetrics(cmap="BuGn", ROC_PR_line="darkorange"),
                test_format_configuration=FormatBinaryClassificationMetrics(cmap="Purples", ROC_PR_line="tab:pink"),
                )
```

## 7. Explanation

```python
trainer.explain_captum(save_dir=TRAIN_EVALUATION_DIR,
                       n_samples=200,
                       n_steps=100)
```

## 8. Save artifacts

```python
# Dataset artifacts
dataset.save_artifacts(TRAIN_ARTIFACTS_DIR)

# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)
inspect_model_architecture(model=model, save_dir=TRAIN_ARTIFACTS_DIR)

# FeatureSchema
schema.to_json(TRAIN_ARTIFACTS_DIR)

# Train log
train_logger(train_config=train_config,
             model_parameters=model_params,
             train_history=history,
             save_directory=TRAIN_ARTIFACTS_DIR.parent)
```

## 9. Finalize Deep Learning

```python
trainer.finalize_model_training(model_checkpoint='current',
                                save_dir=TRAIN_ARTIFACTS_DIR,
                                finalize_config=FinalizeBinaryClassification(filename=train_config.finalized_filename,
                                                                            target_name=dataset.target_names[0],
                                                                            classification_threshold=0.50,
                                                                            class_map=dataset.class_map))
```
