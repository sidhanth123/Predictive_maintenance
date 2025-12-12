import pandas as pd

def load_and_clean_data(csv_path):
    """Loads dataset and applies cleaning + encoding."""
    
    df = pd.read_csv(csv_path)

    # Remove unused columns
    df = df.drop(columns=['UDI', 'Product ID', 'Failure Type'], errors='ignore')

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)

    # Separate features and label
    X = df.drop(columns=['Target'])
    y = df['Target']

    return X, y
