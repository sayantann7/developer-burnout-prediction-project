import os
from dotenv import load_dotenv
import mlflow
import dagshub

load_dotenv()

# Initialize dagshub if credentials exist
if os.getenv("DAGSHUB_REPO_OWNER") and os.getenv("DAGSHUB_REPO_NAME"):
    dagshub.init(repo_owner=os.getenv("DAGSHUB_REPO_OWNER"), repo_name=os.getenv("DAGSHUB_REPO_NAME"), mlflow=True)
else:
    # Use local tracking by default
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Developer_Burnout_Prediction")

from src.logging import logger
from src.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.pipeline.model_trainer import ModelTrainingPipeline
from src.pipeline.model_evaluation import ModelEvaluationPipeline

if __name__ == '__main__':
    try:
        with mlflow.start_run(run_name="Full_Pipeline"):
            STAGE_NAME="DATA INGESTION"
            with mlflow.start_run(run_name=STAGE_NAME, nested=True):
                logger.info(f"---------------------STAGE {STAGE_NAME} INITIATED---------------------")
                data_ingestion_pipeline=DataIngestionTrainingPipeline()
                data_ingestion_pipeline.initiate_data_ingestion()
                logger.info(f"---------------------STAGE STAGE {STAGE_NAME} COMPLETED---------------------")

            STAGE_NAME="DATA TRANSFORMATION"
            with mlflow.start_run(run_name=STAGE_NAME, nested=True):
                logger.info(f"---------------------STAGE {STAGE_NAME} INITIATED---------------------")
                data_transformation_pipeline=DataTransformationTrainingPipeline()
                data_transformation_pipeline.initiate_data_transformation()
                logger.info(f"---------------------STAGE STAGE {STAGE_NAME} COMPLETED---------------------")

            STAGE_NAME="MODEL TRAINING"
            with mlflow.start_run(run_name=STAGE_NAME, nested=True):
                logger.info(f"---------------------STAGE {STAGE_NAME} INITIATED---------------------")
                model_trainer_pipeline=ModelTrainingPipeline()
                model_trainer_pipeline.initiate_model_training()
                logger.info(f"---------------------STAGE STAGE {STAGE_NAME} COMPLETED---------------------")

            STAGE_NAME="MODEL EVALUATION"
            with mlflow.start_run(run_name=STAGE_NAME, nested=True):
                logger.info(f"---------------------STAGE {STAGE_NAME} INITIATED---------------------")
                model_evaluation_pipeline=ModelEvaluationPipeline()
                model_evaluation_pipeline.initiate_model_evaluation()
                logger.info(f"---------------------STAGE STAGE {STAGE_NAME} COMPLETED---------------------")
            
    except Exception as e:
        logger.exception(e)
        raise e