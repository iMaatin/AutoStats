import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import miceforest as mf
from missforest import MissForest
import optuna
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
from sklearn.metrics import mean_squared_error

### Data Preparation Function

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
    df_clean = df.dropna().reset_index(drop=True)
    categorical_cols = []
    discrete_cols = []
    encoders = {}

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            categorical_cols.append(col)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            encoders[col] = le
        else:
            try:
                df_clean[col] = df_clean[col].astype(float)
                if (np.isclose(df_clean[col] % 1, 0).mean() > 0.5):
                    df_clean[col] = df_clean[col].astype(int)
                    discrete_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                encoders[col] = le

    for col in df_clean.columns:
        if col not in categorical_cols and df_clean[col].nunique() == 2:
            categorical_cols.append(col)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            encoders[col] = le

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
        df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
    return df_decoded

def create_missings(df: pd.DataFrame, missingness: float, random_seed: float = 96):
    """
    Create random missingness in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missingness (float): Percentage of missing values to introduce.
        random_seed (float): Seed for reproducibility.
    
    Returns:
        tuple: Original DataFrame, DataFrame with missing values, and a mask DataFrame.
    """
    np.random.seed(random_seed)
    mask = np.random.rand(*df.shape) < (missingness / 100)
    mask_df = pd.DataFrame(mask, columns=df.columns)
    df_missing = df.mask(mask)
    return df, df_missing, mask_df

def simulate_missingness(df, show_missingness=False):
    """
    Simulate missingness by dropping rows with missing values and reintroducing them.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        show_missingness (bool): If True, prints missingness percentages.
    
    Returns:
        tuple: Original DataFrame without missing values, simulated DataFrame with missingness, and a mask.
    """
    missing_original = df.isna().mean()
    df2 = df.dropna().reset_index(drop=True)
    df3 = df2.copy()
    missing_mask = pd.DataFrame(False, index=df3.index, columns=df3.columns)

    for col in df3.columns:
        n_missing = int(round(missing_original[col] * len(df3)))
        if n_missing > 0:
            missing_indices = df3.sample(n=n_missing, random_state=42).index
            df3.loc[missing_indices, col] = np.nan
            missing_mask.loc[missing_indices, col] = True

    if show_missingness:
        missing_df3 = df3.isna().mean()
        print("Missingness Comparison:")
        for col in df.columns:
            print(f"Column '{col}': Original: {missing_original[col]*100:.2f}% \t -> \t df3: {missing_df3[col]*100:.2f}%")

    return df2, df3, missing_mask

### Imputation Functions

def do_knn(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, n_neighbors=5, scale=False):
    """
    Impute missing values using KNN imputation.
    
    Parameters:
        df (pd.DataFrame): DataFrame with missing values.
        continuous_cols (list): Names of continuous numeric columns.
        discrete_cols (list): Names of discrete numeric columns.
        categorical_cols (list): Names of categorical columns.
        n_neighbors (int): Number of neighbors for KNN.
        scale (bool): Whether to apply MinMaxScaler before imputation.
    
    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    df_imputed = df.copy()
    
    if scale:
        scaler = MinMaxScaler()
        df_imputed[continuous_cols] = scaler.fit_transform(df_imputed[continuous_cols])
    
    if continuous_cols:
        imputer_cont = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[continuous_cols] = imputer_cont.fit_transform(df_imputed[continuous_cols])
    
    if discrete_cols:
        imputer_disc = KNNImputer(n_neighbors=n_neighbors)
        imputed_disc = imputer_disc.fit_transform(df_imputed[discrete_cols])
        df_imputed[discrete_cols] = np.round(imputed_disc).astype(int)
    
    if categorical_cols:
        imputer_cat = KNNImputer(n_neighbors=n_neighbors)
        imputed_cat = imputer_cat.fit_transform(df_imputed[categorical_cols])
        df_imputed[categorical_cols] = np.round(imputed_cat).astype(int)
    
    if scale:
        # Reverse scaling for continuous columns
        df_imputed[continuous_cols] = scaler.inverse_transform(df_imputed[continuous_cols])
    
    return df_imputed

def do_mice(df, continuous_cols=None, discrete_cols=None, categorical_cols=None,
            iters=10, strat='normal', scale=False):
    """
    Impute missing values in a DataFrame using the MICE forest method.

    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        continuous_cols (list of str): Names of continuous numeric columns.
        discrete_cols (list of str): Names of discrete numeric columns.
        categorical_cols (list of str): Names of categorical columns.
        iters (int): Number of MICE iterations.
        strat: ['normal', 'shap', 'fast'] or a dictionary specifying the mean matching strategy.
        scale (bool): Whether to apply MinMaxScaler before imputation.

    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    df_imputed = df.copy()

    if scale:
        scaler = MinMaxScaler()
        df_imputed[continuous_cols] = scaler.fit_transform(df_imputed[continuous_cols])

    kernel = mf.ImputationKernel(
        df_imputed,
        random_state=0,
        mean_match_strategy=strat,
        variable_schema=None,  # Explicitly set variable_schema to None 
        )

    kernel.mice(iterations=iters, verbose=False)  # Disable verbose output
    df_completed = kernel.complete_data(dataset=0)

    if discrete_cols:
        df_completed[discrete_cols] = df_completed[discrete_cols].round().astype(int)
    if categorical_cols:
        df_completed[categorical_cols] = df_completed[categorical_cols].round().astype(int)

    if scale:
        scaler = MinMaxScaler()
        df_completed[continuous_cols] = scaler.inverse_transform(df_completed[continuous_cols])

    return df_completed


def do_mf(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, iters=5, scale=False):
    """
    Impute missing values using MissForest.
    
    Parameters:
        df (pd.DataFrame): DataFrame with missing values.
        continuous_cols (list): Names of continuous numeric columns.
        discrete_cols (list): Names of discrete numeric columns.
        categorical_cols (list): Names of categorical columns.
        iters (int): Maximum number of iterations.
        scale (bool): Whether to apply MinMaxScaler before imputation.
    
    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    df_imputed = df.copy()
    
    if scale:
        scaler = MinMaxScaler()
        df_imputed[continuous_cols] = scaler.fit_transform(df_imputed[continuous_cols])
    
    imputer = MissForest(max_iter=iters, categorical=categorical_cols)
    df_imputed_result = imputer.fit_transform(df_imputed)
    
    if discrete_cols:
        df_imputed_result[discrete_cols] = df_imputed_result[discrete_cols].round().astype(int)
    
    if categorical_cols:
        df_imputed_result[categorical_cols] = df_imputed_result[categorical_cols].round().astype(int)
    
    if scale:
        # Reverse scaling for continuous columns
        df_imputed_result[continuous_cols] = scaler.inverse_transform(df_imputed_result[continuous_cols])
    
    return df_imputed_result

### Hyperparameter Optimization with Optuna

def objective(trial, df, continuous_cols, discrete_cols, categorical_cols):
    """
    Objective function for Optuna optimization.
    
    Parameters:
        trial (optuna.Trial): Optuna trial object.
        df (pd.DataFrame): DataFrame with missing values.
        continuous_cols (list): Names of continuous numeric columns.
        discrete_cols (list): Names of discrete numeric columns.
        categorical_cols (list): Names of categorical columns.
    
    Returns:
        float: Evaluation metric (e.g., MSE for continuous columns).
    """
    # Define hyperparameter space
    n_neighbors = trial.suggest_int('n_neighbors', 3, 10)
    iters_mice = trial.suggest_int('iters_mice', 5, 15)
    iters_mf = trial.suggest_int('iters_mf', 3, 10)
    scale_knn = trial.suggest_categorical('scale_knn', [True, False])
    scale_mice = trial.suggest_categorical('scale_mice', [True, False])
    scale_mf = trial.suggest_categorical('scale_mf', [True, False])
    
    # Impute using KNN
    df_knn = do_knn(df, continuous_cols, discrete_cols, categorical_cols, n_neighbors, scale_knn)
    
    # Impute using MICE Forest
    # df_mice = do_mice(df, continuous_cols, discrete_cols, categorical_cols, iters_mice, 'normal', scale_mice)
    
    # Impute using MissForest
    df_mf = do_mf(df, continuous_cols, discrete_cols, categorical_cols, iters_mf, scale_mf)
    
    # Evaluate imputation quality using MSE for continuous columns
    mse_knn = np.mean([mean_squared_error(df[cont_col].dropna(), df_knn[cont_col].dropna()) for cont_col in continuous_cols])
    # mse_mice = np.mean([mean_squared_error(df[cont_col].dropna(), df_mice[cont_col].dropna()) for cont_col in continuous_cols])
    mse_mf = np.mean([mean_squared_error(df[cont_col].dropna(), df_mf[cont_col].dropna()) for cont_col in continuous_cols])
    
    # Use the best imputation method based on MSE
    mse = min(mse_knn, mse_mice, mse_mf)
    
    return mse

def optimize_imputation(df, continuous_cols, discrete_cols, categorical_cols, time_limit=None, min_tries=5):
    """
    Perform hyperparameter optimization for imputation methods using Optuna.
    
    Parameters:
        df (pd.DataFrame): DataFrame with missing values.
        continuous_cols (list): Names of continuous numeric columns.
        discrete_cols (list): Names of discrete numeric columns.
        categorical_cols (list): Names of categorical columns.
        time_limit (float): Time limit in seconds for optimization.
        min_tries (int): Minimum number of trials for each model.
    
    Returns:
        tuple: Best parameters and the best imputed DataFrame.
    """
    study = optuna.create_study(direction='minimize')
    
    if time_limit is not None:
        study.optimize(lambda trial: objective(trial, df, continuous_cols, discrete_cols, categorical_cols), timeout=time_limit, n_trials=min_tries)
    else:
        study.optimize(lambda trial: objective(trial, df, continuous_cols, discrete_cols, categorical_cols), n_trials=min_tries)
    
    best_params = study.best_params
    best_mse = study.best_value
    
    # Re-impute using the best parameters
    df_imputed = None
    if best_mse == np.mean([mean_squared_error(df[cont_col].dropna(), do_knn(df, continuous_cols, discrete_cols, categorical_cols, best_params['n_neighbors'], best_params['scale_knn'])[cont_col].dropna()) for cont_col in continuous_cols]):
        df_imputed = do_knn(df, continuous_cols, discrete_cols, categorical_cols, best_params['n_neighbors'], best_params['scale_knn'])
    elif best_mse == np.mean([mean_squared_error(df[cont_col].dropna(), do_mice(df, continuous_cols, discrete_cols, categorical_cols, best_params['iters_mice'], 'normal', best_params['scale_mice'])[cont_col].dropna()) for cont_col in continuous_cols]):
        df_imputed = do_mice(df, continuous_cols, discrete_cols, categorical_cols, best_params['iters_mice'], 'normal', best_params['scale_mice'])
    else:
        df_imputed = do_mf(df, continuous_cols, discrete_cols, categorical_cols, best_params['iters_mf'], best_params['scale_mf'])
    
    return best_params, df_imputed

### All-in-One Function

def all_in_one_imputation(df, missingness=10, time_limit=300, min_tries=5):
    """
    Perform data preparation, simulate missingness, optimize imputation parameters, and impute missing values.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missingness (float): Percentage of missing values to simulate.
        time_limit (float): Time limit in seconds for optimization.
        min_tries (int): Minimum number of trials for each model.
    
    Returns:
        tuple: Best parameters, imputed DataFrame, and decoded DataFrame.
    """
    # Prepare the dataset
    continuous_cols, discrete_cols, categorical_cols, df_clean, encoders = prep(df)
    
    # Simulate missingness
    _, df_missing, _ = simulate_missingness(df_clean)
    
    # Optimize and perform imputation
    best_params, df_imputed = optimize_imputation(df_missing, continuous_cols, discrete_cols, categorical_cols, time_limit=time_limit, min_tries=min_tries)
    
    # Reverse encoding
    df_imputed_decoded = reverse_encoding(df_imputed, encoders)
    
    return best_params, df_imputed, df_imputed_decoded

# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('train.csv')
    
    # Perform all-in-one imputation
    best_params, df_imputed, df_imputed_decoded = all_in_one_imputation(df, missingness=10, time_limit=300, min_tries=5)
    
    print("Best Parameters:", best_params)
    print("Imputed DataFrame (decoded):")
    print(df_imputed_decoded.head())
