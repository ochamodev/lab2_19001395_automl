import os
import time
import pandas as pd
from joblib import load

def batch_prediction(input_folder, output_folder):
    model = load("model/trained_model.joblib")

    while True:
        for filename in os.listdir(input_folder):
            if filename.endswith(".parquet"):
                input_file_path = os.path.join(input_folder, filename)
                data = pd.read_parquet(input_file_path)
                predictions = model.predict_proba(data)

                pred_df = pd.DataFrame(predictions, columns=[f"Clase{i + 1}" for i in range(predictions.shape[1])])

                output_file_path = os.path.join(output_folder, f"pred_{filename}")
                pred_df.to_parquet(output_file_path)

                os.remove(input_file_path)

                print(f"Procesado y guardado {output_file_path}")
        time.sleep(10)
