
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_column):
    print(df.columns)
    df = df.dropna(axis=0)
    target = df[target_column]
    df = df.drop(columns=[target_column], axis=1).copy()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    encoder = OneHotEncoder()
    encoded_cat = encoder.fit_transform(df[cat_cols])
    feature_names = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_cat)

    df_processed = pd.concat([df[num_cols], encoded_df], axis=1)

    y = target
    return df_processed, y