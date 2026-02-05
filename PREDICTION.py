import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys

df = pd.read_csv("all_stocks_data.csv")

print("CSV Columns Found:\n", df.columns.tolist(), "\n")


def find_col(keywords):
    for col in df.columns:
        for k in keywords:
            if k.lower() in col.lower():
                return col
    return None

cols_map = {
    "PE": find_col(["P/E"]),
    "MarketCap": find_col(["Cap"]),
    "ROCE": find_col(["ROCE"]),
    "SalesGrowth": find_col(["Sales Var"]),
    "ProfitGrowth": find_col(["Profit Var"]),
    "CMP": find_col(["CMP"]),
}

print("Detected Mapping:\n", cols_map, "\n")


if None in cols_map.values():
    print(" Could not detect all required columns. Check names above.")
    sys.exit()


df = df.rename(columns={v: k for k, v in cols_map.items()})


cols = list(cols_map.keys())

for col in cols:
    df[col] = (
        df[col].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=cols)


df["Good_Stock"] = (
    (df["ROCE"] > 20) &
    (df["SalesGrowth"] > 10) &
    (df["ProfitGrowth"] > 10) &
    (df["PE"] < 40)
).astype(int)


X = df[cols]
y = df["Good_Stock"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "multibagger_model.pkl")
    
df["Prediction"] = model.predict(X)
df[df["Prediction"] == 1].to_csv("predicted_multibagger_stocks.csv", index=False)

print("\nDONE. File created: predicted_multibagger_stocks.csv")
