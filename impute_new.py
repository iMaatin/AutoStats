import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import IterativeImputer
import miceforest as mf
from missforest import MissForest
import MIDASpy as md
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def prep(df: pd.DataFrame):
    """
    Preprocess the DataFrame by:
      - Dropping rows with missing values and resetting the index.
      - Converting object columns to categorical via LabelEncoder.
      - Converting other columns to float (and then to int if >50% of values are integer-like).
      - If any numeric column (not already marked as categorical) has only 2 unique values,
        it is considered categorical and encoded.
    
    Returns:
      categorical_cols (list): List of columns encoded as categorical.
      discrete_cols (list): List of columns that are numeric and integer-like.
      cont_cols (list): List of remaining continuous numeric columns.
      df_clean (DataFrame): The preprocessed DataFrame.
      encoders (dict): Mapping from categorical column name to its LabelEncoder.
    """
    # Drop rows with missing values and reset the index.
    df_clean = df.dropna().reset_index(drop=True)

    categorical_cols = []
    discrete_cols = []
    encoders = {}

    # Process each column.
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Mark as categorical and encode using LabelEncoder.
            categorical_cols.append(col)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            encoders[col] = le
        else:
            try:
                # Convert column to float.
                df_clean[col] = df_clean[col].astype(float)
                # If >50% of values are integer-like, cast column to int.
                if (np.isclose(df_clean[col] % 1, 0).mean() > 0.5):
                    df_clean[col] = df_clean[col].astype(int)
                    discrete_cols.append(col)
            except (ValueError, TypeError):
                # If conversion fails, treat the column as categorical.
                categorical_cols.append(col)
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                encoders[col] = le

    # Additionally, if any numeric column (not already marked as categorical) has only 2 unique values,
    # treat it as categorical and encode it.
    for col in df_clean.columns:
        if col not in categorical_cols and df_clean[col].nunique() == 2:
            categorical_cols.append(col)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            encoders[col] = le

    # Continuous columns are those not marked as categorical or discrete.
    continuous_cols = [col for col in df_clean.columns if col not in categorical_cols + discrete_cols]

    return continuous_cols, discrete_cols, categorical_cols, df_clean, encoders

def reverse_encoding(df: pd.DataFrame, encoders: dict):
    """
    Reverse the LabelEncoder transformation on categorical columns.

    Parameters:
      df (pd.DataFrame): DataFrame with encoded categorical columns.
      encoders (dict): Dictionary mapping column names to their LabelEncoder.

    Returns:
      pd.DataFrame: A new DataFrame with the categorical columns decoded to their original labels.
    """
    df_decoded = df.copy()
    for col, le in encoders.items():
        # Ensure that the column is integer type before inverse transforming.
        df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
    return df_decoded

def create_missings(df:pd.DataFrame, missingness:float, random_seed:float=96):
    # Create random missingness.
    np.random.seed(random_seed)
    mask = np.random.rand(*df.shape) < (missingness / 100)
    mask_df = pd.DataFrame(mask, columns=df.columns)
    df_missing = df.mask(mask)
    return df, df_missing, mask_df

def do_knn(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, n_neighbors=5, samples=1):
    """
    Impute missing values in a DataFrame using KNN imputation.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with missing values.
      continuous_cols (list of str): Names of continuous numeric columns.
      discrete_cols (list of str): Names of discrete numeric columns.
      categorical_cols (list of str): Names of categorical columns.
      n_neighbors (int): Number of neighbors for KNN imputation.
      samples (int): Number of imputed DataFrame samples to return.
    
    Returns:
      tuple: (imps, method_info)
          - imps (list of pd.DataFrame): List of imputed DataFrames.
          - method_info (str): String with method information.
    """
    # Work on a copy of the dataframe
    df_imputed = df.copy()
    
    # Create a single imputer for all columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Apply imputation to all columns at once
    imputed_data = imputer.fit_transform(df_imputed)
    df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
    
    # Round discrete and categorical columns to integers
    if discrete_cols:
        df_imputed[discrete_cols] = df_imputed[discrete_cols].round().astype(int)
    if categorical_cols:
        df_imputed[categorical_cols] = df_imputed[categorical_cols].round().astype(int)
    
    # Replicate the imputed DataFrame 'samples' times
    imps = [df_imputed]
    
    # Build method_info string
    method_info = f"KNN, params: n_neighbors={n_neighbors}"
    
    return imps, method_info

def simulate_missingness(df, show_missingness=False):
    """
    Takes a DataFrame, calculates missingness for each column, drops all rows with any missing values (df2),
    then reintroduces missing values to df2 to match the original missingness proportions, resulting in df3.
    Also returns a mask of artificial missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        show_missingness (bool): If True, prints the missingness percentage for each column 
                                 in the original DataFrame and in the simulated DataFrame.
    
    Returns:
        tuple: A tuple (df3, artificial_mask) where:
            - df3 (pd.DataFrame): A new DataFrame with simulated missingness.
            - artificial_mask (pd.DataFrame): A boolean mask indicating the positions where missing values were artificially inserted.
    """
    # 1. Calculate original missingness fraction for each column.
    missing_original = df.isna().mean()
    
    # 2. Drop all rows with missing values to create df2.
    df2 = df.dropna().reset_index(drop=True)
    
    # 3. Create df3 by copying df2.
    df3 = df2.copy()
    
    # Create a mask DataFrame with the same shape as df3 to mark artificial missing values.
    missing_mask = pd.DataFrame(False, index=df3.index, columns=df3.columns)
    
    # 4. Reintroduce missing values in df3 based on the original missingness proportions.
    for col in df3.columns:
        # Calculate the number of entries to set as missing in this column.
        n_missing = int(round(missing_original[col] * len(df3)))
        if n_missing > 0:
            # Randomly select indices to set as missing.
            missing_indices = df3.sample(n=n_missing, random_state=42).index
            df3.loc[missing_indices, col] = np.nan
            missing_mask.loc[missing_indices, col] = True

    # 5. Optionally print missingness for each column.
    if show_missingness:
        missing_df3 = df3.isna().mean()
        print("Missingness Comparison:")
        for col in df.columns:
            print(f"Column '{col}': Original: {missing_original[col]*100:.2f}%  \t -> \t df3: {missing_df3[col]*100:.2f}%")
    
    # Return the simulated DataFrame and the mask.
    return df2, df3, missing_mask 