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


### data prep

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

### imputation functions 
def do_knn(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, n_neighbors=5, samples=1):
    """
    Impute missing values in a DataFrame using KNN imputation.
    
    Assumes:
      - Continuous columns are numeric.
      - Discrete columns are numeric and integer-like.
      - Categorical columns have been label encoded (with missing values represented as np.nan).
    
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
          - method_info (str): String with method information (e.g., "KNN, params: n_neighbors=5").
    """
    # Work on a copy of the dataframe
    df_imputed = df.copy()
    # scaler = MinMaxScaler()
    # scaled_df = scaler.fit_transform(df_imputed)
    # df_imputed = pd.DataFrame(data = scaled_df, columns=df.columns)
    # Impute continuous columns
    if continuous_cols:
        imputer_cont = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[continuous_cols] = imputer_cont.fit_transform(df_imputed[continuous_cols])
    
    # Impute discrete columns and round to integer
    if discrete_cols:
        imputer_disc = KNNImputer(n_neighbors=n_neighbors)
        imputed_disc = imputer_disc.fit_transform(df_imputed[discrete_cols])
        df_imputed[discrete_cols] = np.round(imputed_disc).astype(int)
    
    # Impute categorical columns (assumed to be label encoded)
    if categorical_cols:
        imputer_cat = KNNImputer(n_neighbors=n_neighbors)
        imputed_cat = imputer_cat.fit_transform(df_imputed[categorical_cols])
        df_imputed[categorical_cols] = np.round(imputed_cat).astype(int)
    
    # Replicate the imputed DataFrame 'samples' times.
    imps = df_imputed
    # imps = scaler.inverse_transform(imps)
    # imps = pd.DataFrame(data=imps, columns=df.columns)
    # Build method_info string.
    method_info = f"KNN, params: n_neighbors={n_neighbors}"
    imps = [imps]
    return imps, method_info

def do_scaled_knn(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, n_neighbors=5, samples=1):
    """
    Impute missing values in a DataFrame using KNN imputation.
    
    Assumes:
      - Continuous columns are numeric.
      - Discrete columns are numeric and integer-like.
      - Categorical columns have been label encoded (with missing values represented as np.nan).
    
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
          - method_info (str): String with method information (e.g., "KNN, params: n_neighbors=5").
    """
    # Work on a copy of the dataframe
    df_imputed = df.copy()
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df_imputed)
    df_imputed = pd.DataFrame(data = scaled_df, columns=df.columns)
    imps = scaler.inverse_transform(df_imputed)
    df_imputed = pd.DataFrame(data=imps, columns=df.columns)
    # Impute continuous columns
    if continuous_cols:
        imputer_cont = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[continuous_cols] = imputer_cont.fit_transform(df_imputed[continuous_cols])
    
    # Impute discrete columns and round to integer
    if discrete_cols:
        imputer_disc = KNNImputer(n_neighbors=n_neighbors)
        imputed_disc = imputer_disc.fit_transform(df_imputed[discrete_cols])
        df_imputed[discrete_cols] = np.round(imputed_disc).astype(int)
    
    # Impute categorical columns (assumed to be label encoded)
    if categorical_cols:
        imputer_cat = KNNImputer(n_neighbors=n_neighbors)
        imputed_cat = imputer_cat.fit_transform(df_imputed[categorical_cols])
        df_imputed[categorical_cols] = np.round(imputed_cat).astype(int)
    
    # Replicate the imputed DataFrame 'samples' times.
    # Build method_info string.
    method_info = f"KNN, params: n_neighbors={n_neighbors}"
    imps = [imps]
    return imps, method_info

def do_mice(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, 
                          iters=10, strat='normal', samples=1):
    """
    Impute missing values in a DataFrame using the MICE forest method.
    
    Assumes:
      - Continuous columns are numeric.
      - Discrete columns are numeric and integer-like.
      - Categorical columns have been label encoded (with missing values represented as np.nan).
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with missing values.
      continuous_cols (list of str): Names of continuous numeric columns.
      discrete_cols (list of str): Names of discrete numeric columns.
      categorical_cols (list of str): Names of categorical columns.
      iters (int): Number of MICE iterations.
      strat: ['normal', 'shap', 'fast'] or a dictionary specifying the mean matching strategy.
      samples (int): Number of imputed DataFrame samples to return.
    
    Returns:
      tuple: (imps, method_info)
          - imps (list of pd.DataFrame): List of imputed DataFrames.
          - method_info (str): A string with method information, e.g., 
              "MICE Forest, params: iters=10, strat=shap"
    """
    # Create a copy of the DataFrame to avoid modifying the original data.
    df_imputed = df.copy()
    # scaler = MinMaxScaler()
    # scaled_df = scaler.fit_transform(df_imputed)
    # df_imputed = pd.DataFrame(data = scaled_df, columns=df.columns)
    # Create an imputation kernel using miceforest.
    kernel = mf.ImputationKernel(
        df_imputed,
        random_state=0, 
        mean_match_strategy=strat
    )
    
    # Run the MICE algorithm for the specified number of iterations.
    kernel.mice(iterations=iters)
    
    # Retrieve the completed data (imputed dataset).
    df_completed = kernel.complete_data(dataset=0)
    # df_rev = scaler.inverse_transform(df_completed)
    # df_completed = pd.DataFrame(data=df_rev, columns=df.columns)
    # For discrete and categorical columns, round the imputed values to integers.
    if discrete_cols:
        df_completed[discrete_cols] = df_completed[discrete_cols].round().astype(int)
    if categorical_cols:
        df_completed[categorical_cols] = df_completed[categorical_cols].round().astype(int)
    
    # Replicate the imputed DataFrame 'samples' times.
    imps = [df_completed]
    
    # Build the method info string.
    method_info = f"MICE Forest, params: iters={iters}, strat={strat}"
    
    return imps, method_info


def do_mf(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, iters=5, samples=1):
    """
    Impute missing values in a DataFrame using the MissForest algorithm.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame with missing values.
      continuous_cols (list of str): Names of continuous numeric columns.
      discrete_cols (list of str): Names of discrete numeric columns.
      categorical_cols (list of str): Names of categorical columns.
      iters (int): Maximum number of iterations for the MissForest algorithm.
      samples (int): Number of imputed DataFrame samples to return.
    
    Returns:
      tuple: (imps, method_info)
          - imps (list of pd.DataFrame): List of imputed DataFrames.
          - method_info (str): A string with method information, e.g., "MissForest, params: iters=5"
    """
    df_imputed = df.copy()
    # scaler = MinMaxScaler()
    # scaled_df = scaler.fit_transform(df_imputed)
    # df_imputed = pd.DataFrame(data = scaled_df, columns=df.columns)
    # Create and run the MissForest imputer
    imputer = MissForest(max_iter=iters, categorical=categorical_cols)
    df_imputed_result = imputer.fit_transform(df_imputed)
    # df_rev = scaler.inverse_transform(df_imputed_result)
    # df_imputed_result = pd.DataFrame(data=df_rev, columns=df.columns)
    # For discrete and categorical columns, round the imputed values to integers.
    if discrete_cols:
        df_imputed_result[discrete_cols] = df_imputed_result[discrete_cols].round().astype(int)
    if categorical_cols:
        df_imputed_result[categorical_cols] = df_imputed_result[categorical_cols].round().astype(int)
    
    # Replicate the imputed DataFrame 'samples' times.
    imps = [df_imputed_result]
    
    # Build the method info string.
    method_info = f"MissForest, params: iters={iters}"
    
    return imps, method_info


# def do_midas(df, continuous_cols=None, discrete_cols=None, categorical_cols=None,
#               layer:list=[256,256], vae:bool=True, samples:int=10 ):
#     """
#     Imputes missing values using the MIDAS model.
    
#     Parameters:
#       df (pd.DataFrame): Input dataframe.
#       continuous_cols (list): List of continuous column names.
#       discrete_cols (list): List of discrete (numeric but non-continuous) column names.
#       categorical_cols (list): List of categorical column names.
      
#     Returns:
#       imps (list): A list of imputed dataframes.
#     """
#     # 1. Convert categorical columns and get categorical metadata.
#     md_cat_data, md_cats = md.cat_conv(df[categorical_cols])
    
#     # 2. Define the numeric columns.
#     num_cols = discrete_cols + continuous_cols  # these are the numeric columns

#     # 3. Drop original categorical columns and combine with the converted categorical data.
#     df_copy = df.drop(columns=categorical_cols)
#     constructor_list = [df_copy, md_cat_data]
#     data_in = pd.concat(constructor_list, axis=1)
    
#     # 4. Scale non-categorical columns BEFORE imputation.
#     scaler = StandardScaler()
#     data_in[num_cols] = scaler.fit_transform(data_in[num_cols])
    
#     # (Optional) Handle missing values if needed.
#     na_loc = data_in.isnull()
#     data_in[na_loc] = np.nan
    
#     # 5. Build and train the imputer using the scaled data.
#     imputer = md.Midas(layer_structure=layer, vae_layer=vae, seed=96, input_drop=0.75)
#     # Use md_cats as softmax columns for categorical outputs.
#     imputer.build_model(data_in, softmax_columns=md_cats)
#     imputer.train_model(training_epochs=20)
    
#     # 6. Generate imputations.
#     imps = imputer.generate_samples(m=samples).output_list
    
#     # 7. Post-process each imputed DataFrame.
#     for idx, imp_df in enumerate(imps):
#         # Reverse transform the numeric columns.
#         imp_df[num_cols] = scaler.inverse_transform(imp_df[num_cols])
        
#         # Process categorical columns.
#         # For each softmax group in md_cats, choose the column with the highest probability.
#         tmp_cat = []
#         for group in md_cats:
#             # idxmax returns the column name with maximum value per row for this group.
#             tmp_cat.append(imp_df[group].idxmax(axis=1))
#         # Assume the order of md_cats corresponds to categorical_cols.
#         cat_df = pd.DataFrame({categorical_cols[j]: tmp_cat[j] for j in range(len(categorical_cols))})
        
#         # Drop the softmax columns.
#         flat_cats = [col for group in md_cats for col in group]
#         imp_df = pd.concat([imp_df, cat_df], axis=1).drop(columns=flat_cats)
        
#         # Handle discrete data by rounding the values.
#         imp_df[discrete_cols] = imp_df[discrete_cols].round()
        
#         # Replace the processed DataFrame in the list.
#         imps[idx] = imp_df

#         ### make method info
#         method_info = f'MIDAS, params: samples={samples} ,layer={layer}, vae={vae}'
#     return imps, method_info

def do_midas(df, continuous_cols=None, discrete_cols=None, categorical_cols=None,
              layer:list=[256,256], vae:bool=True, samples:int=10 ):
    """
    Imputes missing values using the MIDAS model.
    
    Parameters:
      df (pd.DataFrame): Input dataframe.
      continuous_cols (list): List of continuous column names.
      discrete_cols (list): List of discrete (numeric but non-continuous) column names.
      categorical_cols (list): List of categorical column names.
      layer (list): Layer configuration for MIDAS.
      vae (bool): Whether to use the VAE layer.
      samples (int): Number of imputed dataframes to return.
      
    Returns:
      tuple: (imps, method_info)
          - imps (list): A list of imputed dataframes.
          - method_info (str): A string with method information.
    """
    # 1. Convert categorical columns and get categorical metadata.
    md_cat_data, md_cats = md.cat_conv(df[categorical_cols])
    
    # 2. Define the numeric columns.
    # Drop categorical columns first.
    df_copy = df.drop(columns=categorical_cols)
    numeric_candidates = discrete_cols + continuous_cols
    num_cols = [col for col in numeric_candidates if col in df_copy.columns]
    
    # 3. Combine numeric data with converted categorical data.
    constructor_list = [df_copy, md_cat_data]
    data_in = pd.concat(constructor_list, axis=1)
    
    # 4. Scale non-categorical columns BEFORE imputation.
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    data_in[num_cols] = scaler.fit_transform(data_in[num_cols])
    
    # (Optional) Handle missing values.
    na_loc = data_in.isnull()
    data_in[na_loc] = np.nan
    
    # 5. Build and train the imputer using the scaled data.
    imputer = md.Midas(layer_structure=layer, vae_layer=vae, seed=96, input_drop=0.75)
    imputer.build_model(data_in, softmax_columns=md_cats)
    imputer.train_model(training_epochs=20)
    
    # 6. Generate imputations.
    imps = imputer.generate_samples(m=samples).output_list
    
    # 7. Post-process each imputed DataFrame.
    for idx, imp_df in enumerate(imps):
        # Reverse transform the numeric columns.
        imp_df[num_cols] = scaler.inverse_transform(imp_df[num_cols])
        
        # Process categorical columns: for each softmax group, pick the max probability column.
        tmp_cat = []
        for group in md_cats:
            tmp_cat.append(imp_df[group].idxmax(axis=1))
        cat_df = pd.DataFrame({categorical_cols[j]: tmp_cat[j] for j in range(len(categorical_cols))})
        
        # Drop the softmax columns.
        flat_cats = [col for group in md_cats for col in group]
        imp_df = pd.concat([imp_df, cat_df], axis=1).drop(columns=flat_cats)
        
        # Handle discrete data by rounding.
        imp_df[discrete_cols] = imp_df[discrete_cols].round()
        
        imps[idx] = imp_df
        
        # Build method info string.
        method_info = f'MIDAS, params: samples={samples} ,layer={layer}, vae={vae}'
    return imps, method_info

#### Select best imputations 

# def select_best_imputations(imputed_dfs, original_df, mask_df, continuous_cols, discrete_cols, categorical_cols, method_info=None, method_names=None):
    
#     n_methods = len(imputed_dfs)
    
#     # If method_info is provided, split it to get the method name and parameters.
#     if method_info is not None:
#         parts = method_info.split(',')
#         extracted_method_name = parts[0].strip()
#         params = ','.join(parts[1:]).strip() if len(parts) > 1 else ""
#         method_names = [f"{extracted_method_name} ({params})"] * n_methods
#     elif method_names is None:
#         method_names = [f"Method {i+1}" for i in range(n_methods)]
    
#     # Dictionary to store best method index per column.
#     best_method_per_col = {}
#     summary_list = []
    
#     # Iterate through each column.
#     for col in original_df.columns:
#         # Determine the data type based on provided lists.
#         if col in continuous_cols:
#             col_data_type = "Continuous"
#         elif col in discrete_cols:
#             col_data_type = "Discrete"
#         elif col in categorical_cols:
#             col_data_type = "Categorical"
#         else:
#             col_data_type = str(original_df[col].dtype)
        
#         # Only evaluate columns that had artificial missing values.
#         if mask_df[col].sum() == 0:
#             best_method_per_col[col] = None
#             summary_list.append({
#                 'Column': col,
#                 'Data Type': col_data_type,
#                 'Best Method': None,
#                 'Metric': np.nan
#             })
#             continue
        
#         # List to store metrics for each method for this column.
#         metrics = []
        
#         if col in continuous_cols or col in discrete_cols:
#             for df_imp in imputed_dfs:
#                 # Convert values to numeric to avoid type errors.
#                 imp_vals = pd.to_numeric(df_imp[col][mask_df[col]], errors='coerce')
#                 orig_vals = pd.to_numeric(original_df[col][mask_df[col]], errors='coerce')
#                 error = np.abs(imp_vals - orig_vals)
#                 mae = error.mean() if not error.empty else np.nan
#                 metrics.append(mae)
#             best_idx = np.nanargmin(metrics)
#             best_metric = metrics[best_idx]
#         elif col in categorical_cols:
#             for df_imp in imputed_dfs:
#                 correct = (df_imp[col][mask_df[col]] == original_df[col][mask_df[col]])
#                 acc = correct.mean() if not correct.empty else np.nan
#                 metrics.append(acc)
#             best_idx = np.nanargmax(metrics)
#             best_metric = metrics[best_idx]
#         else:
#             best_idx = None
#             best_metric = np.nan
        
#         best_method = method_names[best_idx] if best_idx is not None else None
#         best_method_per_col[col] = best_idx
        
#         summary_list.append({
#             'Column': col,
#             'Data Type': col_data_type,
#             'Best Method': best_method,
#             'Metric': best_metric
#         })
    
#     summary_table = pd.DataFrame(summary_list)
    
#     # Build best-imputed dataframe by replacing only the masked entries with the best imputed values.
#     best_imputed_df = original_df.copy()
#     for col in original_df.columns:
#         if mask_df[col].sum() > 0 and best_method_per_col[col] is not None:
#             method_idx = best_method_per_col[col]
#             best_imputed_df.loc[mask_df[col], col] = imputed_dfs[method_idx].loc[mask_df[col], col]
    
#     return best_imputed_df, summary_table

import numpy as np
import pandas as pd

def select_best_imputations(imputed_dfs, original_df, mask_df, continuous_cols, discrete_cols, categorical_cols, method_info=None, method_names=None):
    n_methods = len(imputed_dfs)
    
    # Process method_info to generate method_names if provided.
    if method_info is not None:
        parts = method_info.split(',')
        extracted_method_name = parts[0].strip()
        params = ','.join(parts[1:]).strip() if len(parts) > 1 else ""
        method_names = [f"{extracted_method_name} ({params})"] * n_methods
    elif method_names is None:
        method_names = [f"Method {i+1}" for i in range(n_methods)]
    
    best_method_per_col = {}
    summary_list = []
    
    for col in original_df.columns:
        # Determine the data type label.
        if col in continuous_cols:
            col_data_type = "Continuous"
        elif col in discrete_cols:
            col_data_type = "Discrete"
        elif col in categorical_cols:
            col_data_type = "Categorical"
        else:
            col_data_type = str(original_df[col].dtype)
        
        # Only evaluate columns that had artificial missing values.
        if mask_df[col].sum() == 0:
            best_method_per_col[col] = None
            summary_list.append({
                'Column': col,
                'Data Type': col_data_type,
                'Best Method': None,
                'Metric': np.nan,
                'Error_SD': np.nan,
                'Max_Error': np.nan,
                'Min_Error': np.nan,
                'Within_10pct': np.nan
            })
            continue

        metrics = []
        error_sd = np.nan
        max_error = np.nan
        min_error = np.nan
        within_10pct = np.nan
        
        if col in continuous_cols or col in discrete_cols:
            # Ensure the original column is numeric.
            if not pd.api.types.is_numeric_dtype(original_df[col]):
                raise ValueError(f"Column '{col}' is marked as numeric but contains non-numeric values.")
            for df_imp in imputed_dfs:
                # Convert values to numeric.
                imp_vals = pd.to_numeric(df_imp[col][mask_df[col]], errors='coerce')
                orig_vals = pd.to_numeric(original_df[col][mask_df[col]], errors='coerce')
                errors = np.abs(imp_vals - orig_vals)
                mae = errors.mean() if not errors.empty else np.nan
                metrics.append(mae)
            best_idx = np.nanargmin(metrics)
            best_metric = metrics[best_idx]
            
            # Compute additional metrics for the best method.
            best_imp_vals = pd.to_numeric(imputed_dfs[best_idx][col][mask_df[col]], errors='coerce')
            best_orig_vals = pd.to_numeric(original_df[col][mask_df[col]], errors='coerce')
            errors = np.abs(best_imp_vals - best_orig_vals)
            error_sd = errors.std() if not errors.empty else np.nan
            max_error = errors.max() if not errors.empty else np.nan
            min_error = errors.min() if not errors.empty else np.nan
            
            # Compute fraction within ±10%.
            # For nonzero original values, check error <= 0.1 * |original|.
            # For zeros, require the imputed value to be exactly 0.
            condition = ((best_orig_vals != 0) & (errors <= 0.1 * best_orig_vals.abs())) | \
                        ((best_orig_vals == 0) & (errors == 0))
            within_10pct = condition.mean() if not condition.empty else np.nan
            
        elif col in categorical_cols or pd.api.types.is_string_dtype(original_df[col]):
            # For categorical columns, compute accuracy.
            for df_imp in imputed_dfs:
                correct = (df_imp[col][mask_df[col]] == original_df[col][mask_df[col]])
                acc = correct.mean() if not correct.empty else np.nan
                metrics.append(acc)
            best_idx = np.nanargmax(metrics)
            best_metric = metrics[best_idx]
            # Extra metrics are not applicable for categoricals.
            error_sd = np.nan
            max_error = np.nan
            min_error = np.nan
            within_10pct = np.nan
        else:
            best_idx = None
            best_metric = np.nan
        
        best_method = method_names[best_idx] if best_idx is not None else None
        best_method_per_col[col] = best_idx
        
        summary_list.append({
            'Column': col,
            'Data Type': col_data_type,
            'Best Method': best_method,
            'Metric': best_metric,
            'Error_SD': error_sd,
            'Max_Error': max_error,
            'Min_Error': min_error,
            'Within_10pct': within_10pct
        })
    
    summary_table = pd.DataFrame(summary_list)
    
    # Build best-imputed DataFrame by replacing masked entries with values from the best method.
    best_imputed_df = original_df.copy()
    for col in original_df.columns:
        if mask_df[col].sum() > 0 and best_method_per_col[col] is not None:
            method_idx = best_method_per_col[col]
            best_imputed_df.loc[mask_df[col], col] = imputed_dfs[method_idx].loc[mask_df[col], col]
    
    return best_imputed_df, summary_table


###### aio pipeline function

def aio_custom_missingness(df, missingness_percent=20,
                                        knn_neighbors=5,
                                        mice_iters=10, mice_strat='normal',
                                        mf_iters=5,
                                        midas_layer=[256,256], midas_vae=True):
    """
    End-to-end pipeline to:
      1. Preprocess the DataFrame.
      2. Create artificial missingness.
      3. Impute the missing values using KNN, MICE Forest, MissForest, and MIDAS.
      4. Select the best imputation for each column based on column-specific metrics.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      missingness_percent (float): Percentage of missingness to introduce.
      knn_neighbors (int): Parameter for KNN imputation.
      mice_iters (int): Number of iterations for MICE forest.
      mice_strat (str): Mean matching strategy for MICE forest.
      mf_iters (int): Number of iterations for MissForest.
      midas_layer (list): Layer configuration for MIDAS.
      midas_vae (bool): Whether to use the VAE layer in MIDAS.
      
    Returns:
      best_imputed (pd.DataFrame): DataFrame with best imputed values for each column.
      summary_table (pd.DataFrame): Summary table showing, per column, the best method and its metric.
    """
    # 1. Preprocess the DataFrame.
    continuous_cols, discrete_cols, categorical_cols, df_clean, encoders = prep(df)
    
    # 2. Create artificial missingness.
    # Here, simulate_missingness returns (df_complete, df_missing, missing_mask)
    df_complete, df_missing, missing_mask = create_missings(df_clean, missingness=missingness_percent)
    
    # 3. Run all imputation methods on the DataFrame with missing values.
    knn_imps, knn_info = do_knn(df_missing, continuous_cols, discrete_cols, categorical_cols,
                                n_neighbors=knn_neighbors, samples=1)
    mice_imps, mice_info = do_mice(df_missing, continuous_cols, discrete_cols, categorical_cols,
                                   iters=mice_iters, strat=mice_strat, samples=1)
    mf_imps, mf_info = do_mf(df_missing, continuous_cols, discrete_cols, categorical_cols,
                             iters=mf_iters, samples=1)
    midas_imps, midas_info = do_midas(df_missing, continuous_cols, discrete_cols, categorical_cols,
                                      layer=midas_layer, vae=midas_vae, samples=1)
    
    # Each method returns a list with one imputed DataFrame.
    imputed_dfs = [knn_imps[0], mice_imps[0], mf_imps[0], midas_imps[0]]
    # Build a list of method info strings (one per method).
    method_infos = [knn_info, mice_info, mf_info, midas_info]
    
    # 4. Use the select_best_imputations function to choose the best imputation for each column.
    best_imputed, summary_table = select_best_imputations(imputed_dfs, df_complete, missing_mask,
                                                          continuous_cols, discrete_cols, categorical_cols,
                                                          method_names=method_infos)
    
    decoded = reverse_encoding(best_imputed, encoders)
    return decoded, summary_table, df_missing, missing_mask

def aio_simulated_missingness(df, knn_neighbors=5,
                                  mice_iters=10, mice_strat='normal',
                                  mf_iters=5,
                                  midas_layer=[256,256], midas_vae=True):
    """
    End-to-end pipeline to:
      1. Preprocess the DataFrame.
      2. Create artificial missingness.
      3. Impute the missing values using KNN, MICE Forest, MissForest, and MIDAS.
      4. Select the best imputation for each column based on column-specific metrics.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame.
      missingness_percent (float): Percentage of missingness to introduce.
      knn_neighbors (int): Parameter for KNN imputation.
      mice_iters (int): Number of iterations for MICE forest.
      mice_strat (str): Mean matching strategy for MICE forest.
      mf_iters (int): Number of iterations for MissForest.
      midas_layer (list): Layer configuration for MIDAS.
      midas_vae (bool): Whether to use the VAE layer in MIDAS.
      
    Returns:
      best_imputed (pd.DataFrame): DataFrame with best imputed values for each column.
      summary_table (pd.DataFrame): Summary table showing, per column, the best method and its metric.
    """
    # 1. Preprocess the DataFrame.
    continuous_cols, discrete_cols, categorical_cols, df_clean, encoders = prep(df)
    
    # 2. Create artificial missingness.
    # Here, simulate_missingness returns (df_complete, df_missing, missing_mask)
    df_complete, df_missing, missing_mask = simulate_missingness(df_clean, show_missingness=True)
    
    # 3. Run all imputation methods on the DataFrame with missing values.
    knn_imps, knn_info = do_knn(df_missing, continuous_cols, discrete_cols, categorical_cols,
                                n_neighbors=knn_neighbors, samples=1)
    mice_imps, mice_info = do_mice(df_missing, continuous_cols, discrete_cols, categorical_cols,
                                   iters=mice_iters, strat=mice_strat, samples=1)
    mf_imps, mf_info = do_mf(df_missing, continuous_cols, discrete_cols, categorical_cols,
                             iters=mf_iters, samples=1)
    midas_imps, midas_info = do_midas(df_missing, continuous_cols, discrete_cols, categorical_cols,
                                      layer=midas_layer, vae=midas_vae, samples=1)
    
    # Each method returns a list with one imputed DataFrame.
    imputed_dfs = [knn_imps[0], mice_imps[0], mf_imps[0], midas_imps[0]]
    # Build a list of method info strings (one per method).
    method_infos = [knn_info, mice_info, mf_info, midas_info]
    
    # 4. Use the select_best_imputations function to choose the best imputation for each column.
    best_imputed, summary_table = select_best_imputations(imputed_dfs, df_complete, missing_mask,
                                                          continuous_cols, discrete_cols, categorical_cols,
                                                          method_names=method_infos)
    decoded = reverse_encoding(best_imputed, encoders)

    return decoded, summary_table


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

