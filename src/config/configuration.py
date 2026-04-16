from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml, create_directories
from src.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_ckpt=Path(config.model_ckpt),

            n_trials=params.optuna.n_trials,
            direction=params.optuna.direction,

            models=params.model_selection.models,

            params={
                k: v for k, v in params.items()
                if k not in ["model_selection", "optuna"]
            }
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name)
        )