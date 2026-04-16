import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

from src.logging import logger
from src.utils.common import create_directories
from src.utils.model_utils import tune_model

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        create_directories([self.config.root_dir])

    def train(self):
        logger.info("Starting model training process...")

        logger.info("Loading training and testing data...")

        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        X_train = train_df.drop("burnout_level", axis=1)
        y_train = train_df["burnout_level"]

        X_test = test_df.drop("burnout_level", axis=1)
        y_test = test_df["burnout_level"]

        logger.info("Data loaded successfully.")

        result = tune_model(X_train, y_train, self.config)

        best_model = result["model"]

        logger.info(f"Best model: {result['model_type']}")
        logger.info(f"Best params: {result['best_params']}")
        logger.info(f"Best CV score: {result['best_score']:.4f}")

        logger.info("Evaluating model on test data...")

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")

        logger.info("Saving trained model...")

        joblib.dump(best_model, self.config.model_ckpt)

        logger.info(f"Model saved at {self.config.model_ckpt}")

        return accuracy