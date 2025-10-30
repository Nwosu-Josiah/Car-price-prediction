import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from loguru import logger

class DataPreprocessor:

    def __init__(self, max_categories: int = 50):
        # Limit rare categories to prevent feature explosion
        self.max_categories = max_categories
        self.encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=True,  # keep sparse matrix
            max_categories=self.max_categories
        )
        logger.info(f"Initialized OneHotEncoder with max_categories={self.max_categories}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with missing target and duplicates.
        """
        initial_shape = df.shape
        df = df.dropna(subset=['price']).drop_duplicates()
        logger.info(f"Cleaned data: {initial_shape} → {df.shape}")
        return df

    def encode_features(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """
        Encodes categorical columns using OneHotEncoder.
        Keeps output sparse to avoid large memory consumption.
        """
        cat_cols = ['manufacturer', 'fuel', 'transmission', 'model', 'condition']
        df = df.copy()

        # Handle unseen columns gracefully
        for col in cat_cols:
            if col not in df.columns:
                df[col] = "Unknown"

        if fit:
            encoded = self.encoder.fit_transform(df[cat_cols])
        else:
            encoded = self.encoder.transform(df[cat_cols])

        # Create sparse DataFrame
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded,
            columns=self.encoder.get_feature_names_out(cat_cols),
        )

        # Merge sparse + numeric data efficiently
        non_cat_df = df.drop(columns=cat_cols)
        result_df = pd.concat([non_cat_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        logger.info(f"Encoded {len(cat_cols)} categorical columns → {encoded.shape[1]} encoded features.")
        return result_df
