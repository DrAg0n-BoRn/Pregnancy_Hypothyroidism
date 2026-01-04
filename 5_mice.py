from ml_tools.MICE import DragonMICE
from ml_tools.schema import FeatureSchema

from paths import PM


if __name__ == "__main__":
    feature_schema = FeatureSchema.from_json(PM.engineering_artifacts)
    
    imputer = DragonMICE(schema=feature_schema,
                         impute_targets=False,
                         iterations=30,
                         resulting_datasets=3)
    
    imputer.run_pipeline(df_path_or_dir=PM.engineering_data_file,
                         save_datasets_dir=PM.mice_datasets,
                         save_metrics_dir=PM.mice_results)
