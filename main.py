import os
from dotenv import load_dotenv
import pandas as pd
from api_mode import api
from batch_mode import batch
import model_train
import preprocess

load_dotenv()



def main():
    DATASET_PATH = os.getenv('DATASET')
    TARGET = os.getenv('TARGET')
    MODEL = os.getenv('MODEL')
    TRIALS = int(os.getenv('TRIALS'))
    DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE")
    PORT = 0

    data = pd.read_parquet(DATASET_PATH)

    x, y = preprocess.preprocess_data(data, TARGET)

    model_train.optimize_and_train(x, y, MODEL, TRIALS)

    if DEPLOYMENT_TYPE == 'Batch':
        INPUT_FOLDER = os.getenv('INPUT_FOLDER')
        OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER')
        print("Executing batch prediction...")
        batch.batch_prediction(INPUT_FOLDER, OUTPUT_FOLDER)
    elif DEPLOYMENT_TYPE == 'API':
        PORT = int(os.getenv('PORT'))
        print("Executing API")
        api.start_api(PORT)
    else:
        print(f"""
        Unsupported deployment type: {DEPLOYMENT_TYPE}.
        Please choose one of the following: Batch, API
        """)

    print(f"deployment_type={DEPLOYMENT_TYPE}")

if __name__ == "__main__":
    main()