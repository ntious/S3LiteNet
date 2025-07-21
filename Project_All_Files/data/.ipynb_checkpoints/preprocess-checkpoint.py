
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from config import DATASET_PATH

def load_and_preprocess():
    df = pd.read_csv(DATASET_PATH)
    df.drop(columns=["Src IP", "Dst IP", "Flow ID", "Timestamp", "Attack Name"], inplace=True, errors='ignore')
    X = df.drop(columns=["Label"])
    y = df["Label"]

    from config import RANDOM_STATE, TEST_SIZE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
