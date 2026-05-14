import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath, fill_method='mean'):
    """
    Load dataset and handle zeros in clinical columns.
    """
    df = pd.read_csv(filepath)
    
    # Columns where 0 is likely a missing value
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in cols_to_fix:
        if fill_method == 'mean':
            fill_val = df[df[col] != 0][col].mean()
        else:
            fill_val = df[df[col] != 0][col].median()
        df[col] = df[col].replace(0, fill_val)
        
    return df

def get_train_test_data(df, target_col='Outcome', test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
