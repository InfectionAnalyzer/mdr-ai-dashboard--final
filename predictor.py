import pandas as pd
import joblib

def predict_objective(input_df, model_file):
    model = joblib.load(model_file)
    input_df_encoded = input_df.copy()
    for col in input_df_encoded.select_dtypes(include=['object', 'category']).columns:
        input_df_encoded[col] = input_df_encoded[col].astype('category').cat.codes
    predictions = model.predict(input_df_encoded)
    proba = model.predict_proba(input_df_encoded) if hasattr(model, 'predict_proba') else None
    results = pd.DataFrame(predictions, columns=["Prediction"])
    if proba is not None:
        proba_df = pd.DataFrame(proba, columns=[f"Prob_{i}" for i in range(proba.shape[1])])
        results = pd.concat([results, proba_df], axis=1)
    return results
