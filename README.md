# predicting-multibagger-stocks-using-machine-learning


This project is a Multibagger Stock Predictor built using Python, Pandas, and Scikit-learn. It analyzes stock data from a CSV file, cleans and processes financial indicators, and applies a RandomForestClassifier to identify potential "multibagger" stocks—companies that show strong fundamentals and could deliver high returns in the future.
The model is trained on features like P/E ratio, Market Cap, ROCE, Sales Growth, Profit Growth, and CMP, and then classifies stocks as either "Good" or "Not Good" based on predefined financial rules and machine learning predictions.



⚡ Brief Explanation
- Data Loading & Inspection
- Reads stock data from a CSV file.
- Detects and maps important financial columns automatically.
- Data Cleaning
- Removes unwanted characters (commas, %).
- Converts values to numeric format.
- Drops rows with missing values.
- Label Creation
- Defines a "Good Stock" if ROCE > 20, Sales Growth > 10%, Profit Growth > 10%, and PE < 40.
- Model Training
- Splits data into training and testing sets.
- Trains a RandomForest model with 200 trees.
- Evaluates accuracy using accuracy_score.
- Model Saving & Prediction
- Saves the trained model (multibagger_model.pkl) using joblib.
- Predicts on the full dataset.
- Outputs a CSV (predicted_multibagger_stocks.csv) containing only the predicted good stocks.
