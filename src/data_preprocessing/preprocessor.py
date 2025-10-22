import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['price'])
        df = df.drop_duplicates()
        return df

    def encode_features(self, df: pd.DataFrame, fit=False):
        cat_cols = ['brand', 'fuel_type', 'transmission']
        if fit:
            encoded = self.encoder.fit_transform(df[cat_cols])
        else:
            encoded = self.encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded.toarray(), columns=self.encoder.get_feature_names_out(cat_cols))
        return pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
