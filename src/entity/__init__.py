from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessor_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_ckpt: Path
    n_trials: int
    direction: str
    models: List[str]
    params: Dict[str, Any]
    
@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    metric_file_name: Path