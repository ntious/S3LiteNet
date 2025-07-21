from config import DATASET_PATH, RANDOM_STATE, TEST_SIZE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess():
    """Dispatch to appropriate dataset-specific preprocessing based on path name."""
    path = DATASET_PATH.lower()

    if "cic_iot_diad" in path:
        return process_diad_dataset()
    elif "ton_iot_modbus" in path:
        return process_toniot_modbus_dataset()
    elif "ton_iot_thermostat" in path:
        return process_toniot_thermostat_dataset()
    else:
        raise ValueError(f"Unrecognized dataset in path: {DATASET_PATH}")

def preprocess_common(df, label_col="Label"):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def process_diad_dataset():
    df = pd.read_csv(DATASET_PATH)
    df.drop(columns=["Src IP", "Dst IP", "Flow ID", "Timestamp", "Attack Name"], inplace=True, errors='ignore')
    return preprocess_common(df)


def process_toniot_thermostat_dataset():
    # Read as single column
    df = pd.read_csv(DATASET_PATH, header=None)

    # Split single string into multiple columns and skip the first row (which is just header again)
    df = df.iloc[:, 0].str.split(",", expand=True)
    df = df.iloc[1:]  # Skip the row with string headers

    # Assign proper column names manually
    df.columns = ['date', 'time', 'current_temperature', 'thermostat_status', 'Label', 'type']
    df.reset_index(drop=True, inplace=True)
    # Drop unnecessary columns
    df.drop(columns=["date", "time"], inplace=True, errors='ignore')

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(0)

    # Ensure Label is integer
    df['Label'] = df['Label'].astype(int)

    # One-hot encode if 'type' exists
    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

    return preprocess_common(df)


def process_toniot_modbus_dataset():
    df = pd.read_csv(DATASET_PATH)
    df.drop(columns=["date", "time", "Flow ID", "Timestamp"], inplace=True, errors='ignore')
    #Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(0)
    #Ensure labels are binary integers (if not already)
    df['Label'] = df['Label'].astype(int)
    #Encode categorical columns (like type) if needed
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    return preprocess_common(df)
