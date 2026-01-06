from ml_tools.path_manager import DragonPathManager


PM = DragonPathManager(anchor_file=__file__,
                       base_directories=["data", "results", "raw_data", "helpers", "backups"])


# Directories
PM.clean_data = PM.data / "Clean Data"
PM.engineering = PM.data / "Feature Engineering"
PM.vif = PM.data / "VIF"
PM.mice_datasets = PM.data / "MICE Datasets"
PM.mice_results = PM.results / "MICE Metrics"
PM.train_datasets = PM.data / "Train Datasets"
PM.resampling = PM.data / "Resampling"

# Subdirectories
PM.engineering_plots = PM.engineering / "Plots"
PM.engineering_artifacts = PM.engineering / "Artifacts"
PM.engineering_datasets = PM.engineering / "Datasets"

# Files
PM.raw_data_file = PM.raw_data / "raw_data.csv"
PM.raw_targets_file = PM.raw_data / "raw_targets.csv"
PM.feature_columns_file = PM.raw_data / "feature_columns.csv"
PM.translation_file = PM.raw_data / "translation.json"

PM.processed_targets_file = PM.clean_data / "processed_targets.csv"
PM.clean_data_cn_file = PM.clean_data / "clean_data_cn.csv"
PM.processed_data_cn_file = PM.clean_data / "processed_data_cn.csv"
PM.processed_data_file = PM.clean_data / "processed_data.csv"

PM.engineering_data_file = PM.engineering_datasets / "engineered_data.csv"


if __name__ == "__main__":
    PM.make_dirs()
    PM.status()
