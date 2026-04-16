## VERY IMPORTANT
Must download the data here:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv

# Download "phishing_email.csv" and move to data/

## Pipeline

# datapipeline.py
This loads the data and vectorizes into splits (80/10/10)

# training files: neural network + xgboost model
Two separate models for this project
- neural network 256 -> 64 -> sigmoid
- xgboost: 200 trees, 5 depth, 0.1 learning rate

# evaluate.py
Metrics for evaluation for each model and also confusion matrices for false/true negatives/positives

# predict.py
Predictions using the saved models

# app.py
This is prototype interface so we can enter in email body (text) and use the models to predict

Run with
python src/data_pipeline.py && python src/train_nn.py && python src/train_xgb.py && python src/evaluate.py

- only need to run data_pipeline.py once
- only need to run training files once (unless we change architecture)
