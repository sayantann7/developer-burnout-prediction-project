from src.logging import logger
from src.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.data_transformation import DataTransformationTrainingPipeline

STAGE_NAME="DATA INGESTION"

try:
    logger.info(f"---------------------STAGE {STAGE_NAME} INITIATED---------------------")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"---------------------STAGE STAGE {STAGE_NAME} COMPLETED---------------------")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="DATA TRANSFORMATION"

try:
    logger.info(f"---------------------STAGE {STAGE_NAME} INITIATED---------------------")
    data_transformation_pipeline=DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"---------------------STAGE STAGE {STAGE_NAME} COMPLETED---------------------")
except Exception as e:
    logger.exception(e)
    raise e