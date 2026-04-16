import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.entity import DataTransformationConfig
from src.utils.common import create_directories

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir, self.config.preprocessor_path])

    def transform_and_save(self):
        data = pd.read_csv(self.config.data_path)

        data = data.dropna(subset=["burnout_level"])
        num_cols = data.select_dtypes(include='number')
        data[num_cols.columns] = num_cols.fillna(num_cols.mean())

        # Encode target variable
        le = LabelEncoder()
        data["burnout_level"] = le.fit_transform(data["burnout_level"])
        
        # Save LabelEncoder
        le_path = f"{self.config.preprocessor_path}/label_encoder.pkl"
        joblib.dump(le, le_path)

        train_df, test_df = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=data["burnout_level"]
        )

        train_df.to_csv(f"{self.config.root_dir}/train.csv", index=False)
        test_df.to_csv(f"{self.config.root_dir}/test.csv", index=False)

