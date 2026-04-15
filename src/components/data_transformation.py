import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity import DataTransformationConfig
from src.utils.common import create_directories

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir])

    def transform_and_save(self):
        data = pd.read_csv(self.config.data_path)

        data = data.dropna(subset=["burnout_level"])
        num_cols = data.select_dtypes(include='number')
        data[num_cols.columns] = num_cols.fillna(num_cols.mean())

        train_df, test_df = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=data["burnout_level"]
        )

        train_df.to_csv(f"{self.config.root_dir}/train.csv", index=False)
        test_df.to_csv(f"{self.config.root_dir}/test.csv", index=False)

