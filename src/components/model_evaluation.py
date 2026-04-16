import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logging import logger
from src.utils.common import create_directories

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        create_directories([self.config.root_dir])

    def evaluate(self):
        logger.info("Starting model evaluation...")

        # load model
        logger.info("Loading trained model...")
        model = joblib.load(self.config.model_path)

        # test data
        logger.info("Loading test data...")
        test_df = pd.read_csv(f"{self.config.data_path}/test.csv")

        X_test = test_df.drop("burnout_level", axis=1)
        y_test = test_df["burnout_level"]

        logger.info("Data loaded successfully.")

        # predictions
        logger.info("Generating predictions...")
        y_pred = model.predict(X_test)

        # metrics
        logger.info("Calculating evaluation metrics...")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # confusion matrix
        logger.info("Generating confusion matrix...")

        cm = confusion_matrix(y_test, y_pred)

        le = joblib.load(f"{self.config.data_path}/preprocessor/label_encoder.pkl")
        labels = le.classes_

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = f"{self.config.root_dir}/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        logger.info(f"Confusion matrix saved at {cm_path}")

        logger.info(f"Metrics: {metrics}")

        # save metrics
        logger.info("Saving metrics...")

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.config.metric_file_name, index=False)

        logger.info(f"Metrics saved at {self.config.metric_file_name}")

        return metrics