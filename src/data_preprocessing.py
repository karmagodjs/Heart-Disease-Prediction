
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


DATA_PATH = "dataset/heart.csv"
TARGET = "num"
DROP_COLUMNS = ["id", "dataset"]
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        na_values=["?", "NA", "N/A", ""]
    )
    print("Original Dataset Shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = (df[TARGET] > 0).astype(int)

    print("\nTarget Distribution:")
    print(df[TARGET].value_counts())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFinal Dataset Shape:", df.shape)

    if df[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both target classes.")
    return df

def prepare_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    categorical = [
        col for col in CATEGORICAL_COLUMNS
        if col in X.columns
    ]
    categorical += [
        col for col in X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        if col not in categorical
    ]
    numerical = [
        col for col in X.columns
        if col not in categorical
    ]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        numerical,
        categorical
    )