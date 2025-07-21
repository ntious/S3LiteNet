import pandas as pd

# Reload the datasets after kernel reset
#thermostat_df = pd.read_csv("IoT_Thermostat.csv")
modbus_df = pd.read_csv("TON_IoT_Modbus.csv")
thermostat_df = pd.read_csv("TON_IoT_Thermostat.csv", dtype={19: str})

# Summarize both datasets
summary = {
    "Thermostat": {
        "Shape": thermostat_df.shape,
        "Columns": thermostat_df.columns.tolist(),
        "Missing Values": thermostat_df.isnull().sum().to_dict(),
        "Unique Labels": thermostat_df['label'].unique().tolist() if 'label' in thermostat_df else []
    },
    "Modbus": {
        "Shape": modbus_df.shape,
        "Columns": modbus_df.columns.tolist(),
        "Missing Values": modbus_df.isnull().sum().to_dict(),
        "Unique Labels": modbus_df['label'].unique().tolist() if 'label' in modbus_df else []
    }
}

print (summary)
