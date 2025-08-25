### Imports 
import warnings
warnings.filterwarnings("default")  # keep default until pipeline runs

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
import MIDASpy as md

def prep(df: pd.DataFrame):
    """
    Preprocess the DataFrame by:
    - Dropping rows with missing values and resetting the index.
    - Converting object columns to categorical via LabelEncoder.
    - Converting other columns to float (and then to int if >50% of values are integer-like).
    - If any numeric column (not already marked as categorical) has only 2 unique values,
      it is considered categorical and encoded.

    Returns:
        continuous_cols (list): List of remaining continuous numeric columns.
        discrete_cols (list): List of columns that are numeric and integer-like.
        categorical_cols (list): List of columns encoded as categorical.
        df_clean (DataFrame): The preprocessed DataFrame.
        encoders (dict): Mapping from categorical column name to its LabelEncoder.
    """
    # Drop rows with missing values.
    df_clean = df.dropna().reset_index(drop=True)
    categorical_cols = []
    discrete_cols = []

    # Loop over each column to check its type and convert accordingly.
    for col in df_clean.columns:
        # If the column type is object, encode it as a categorical variable.
        if df_clean[col].dtype == 'object' or df_clean[col].nunique() == 2:
            categorical_cols.append(col)
        else:
            try:
                # Convert column to float first.
                df_clean[col] = df_clean[col].astype(float)
                # Check if most of the values are integer-like using np.isclose.
                # This computes the proportion of values where the modulus with 1 is nearly 0.
                if (np.isclose(df_clean[col] % 1, 0)).mean() > 0.5:
                    df_clean[col] = df_clean[col].astype(int)
                    discrete_cols.append(col)
            except (ValueError, TypeError):
                # If conversion to float fails, treat the column as categorical.
                categorical_cols.append(col)
                

    # Determine continuous columns as those not flagged as categorical or discrete.
    continuous_cols = [col for col in df_clean.columns if col not in categorical_cols + discrete_cols]

    return continuous_cols, discrete_cols, categorical_cols

def simulate_missingness(df, show_missingness=False, random_state=42):
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
            missing_indices = df3.sample(n=n_missing, random_state=random_state).index
            df3.loc[missing_indices, col] = np.nan
            missing_mask.loc[missing_indices, col] = True

    if show_missingness:
        missing_df3 = df3.isna().mean()
        print("Missingness Comparison:")
        for col in df.columns:
            print(f"Column '{col}': Original: {missing_original[col]*100:.2f}% \t -> \t df3: {missing_df3[col]*100:.2f}%")

    return df2, df3, missing_mask

def create_missings(df: pd.DataFrame, missingness: float, random_seed: float = 96):
    """
    Create random missingness in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        missingness (float): Percentage of missing values to introduce.
        random_seed (float): Seed for reproducibility.
    
    Returns:
        tuple: (original DataFrame, DataFrame with missing values, mask DataFrame)
    """
    np.random.seed(random_seed)
    mask = np.random.rand(*df.shape) < (missingness / 100)
    mask_df = pd.DataFrame(mask, columns=df.columns)
    df_missing = df.mask(mask)
    return df, df_missing, mask_df


def do_knn(df, continuous_cols=None, discrete_cols=None, categorical_cols=None, n_neighbors=5, scale=False):
    """
    Impute missing values using KNN imputation over all columns.

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
    encoders = {}

    # Encode categorical columns
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            not_null = df_imputed[col].dropna()
            if not not_null.empty:
                le.fit(not_null)
                df_imputed[col] = df_imputed[col].map(lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
                encoders[col] = le
            else:
                # All values missing in this column
                encoders[col] = None

    # Optionally scale numeric columns
    if scale:
        scaler = MinMaxScaler()
        df_imputed[df.columns] = scaler.fit_transform(df_imputed)

    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed[df.columns] = imputer.fit_transform(df_imputed)

    # Reverse scale
    if scale:
        df_imputed[df.columns] = scaler.inverse_transform(df_imputed)

    # Round discrete and categorical values
    if discrete_cols:
        df_imputed[discrete_cols] = np.round(df_imputed[discrete_cols]).astype(int)
    if categorical_cols:
        for col in categorical_cols:
            df_imputed[col] = np.round(df_imputed[col]).astype(int)
            if encoders[col] is not None:
                inv_map = dict(enumerate(encoders[col].classes_))
                df_imputed[col] = df_imputed[col].map(inv_map)

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
    encoders = {}

    # Encode categorical columns
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            not_null = df_imputed[col].dropna()
            if not not_null.empty:
                le.fit(not_null)
                df_imputed[col] = df_imputed[col].map(lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
                encoders[col] = le
            else:
                encoders[col] = None

    # Scale continuous columns if requested
    if scale and continuous_cols:
        scaler = MinMaxScaler()
        df_imputed[continuous_cols] = scaler.fit_transform(df_imputed[continuous_cols])

    # Run MICE imputation
    kernel = mf.ImputationKernel(
        df_imputed,
        random_state=0,
        mean_match_strategy=strat,
        variable_schema=None
    )

    kernel.mice(iterations=iters, verbose=False)
    df_completed = kernel.complete_data(dataset=0)

    # Post-process discrete and categorical columns
    if discrete_cols:
        df_completed[discrete_cols] = df_completed[discrete_cols].round().astype(int)

    if categorical_cols:
        for col in categorical_cols:
            df_completed[col] = np.round(df_completed[col]).astype(int)
            if encoders[col] is not None:
                inv_map = dict(enumerate(encoders[col].classes_))
                df_completed[col] = df_completed[col].map(inv_map)

    # Reverse scaling
    if scale and continuous_cols:
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
    encoders = {}

    # Encode categorical columns
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            not_null = df_imputed[col].dropna()
            if not not_null.empty:
                le.fit(not_null)
                df_imputed[col] = df_imputed[col].map(lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
                encoders[col] = le
            else:
                encoders[col] = None

    # Scale continuous columns
    if scale and continuous_cols:
        scaler = MinMaxScaler()
        df_imputed[continuous_cols] = scaler.fit_transform(df_imputed[continuous_cols])

    # Impute with MissForest
    imputer = MissForest(max_iter=iters)
    df_imputed_result = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df.columns)

    # Post-process discrete columns
    if discrete_cols:
        df_imputed_result[discrete_cols] = df_imputed_result[discrete_cols].round().astype(int)

    # Post-process categorical columns
    if categorical_cols:
        for col in categorical_cols:
            df_imputed_result[col] = df_imputed_result[col].round().astype(int)
            if encoders[col] is not None:
                inv_map = dict(enumerate(encoders[col].classes_))
                df_imputed_result[col] = df_imputed_result[col].map(inv_map)

    # Reverse scaling
    if scale and continuous_cols:
        df_imputed_result[continuous_cols] = scaler.inverse_transform(df_imputed_result[continuous_cols])

    return df_imputed_result

def do_midas(df,
             continuous_cols=None,
             discrete_cols=None,
             categorical_cols=None,
             layer: list = [256, 256],
             vae: bool = True,
             samples: int = 10,
             input_drop: float = 0.75,
             training_epochs: int = 20,
             random_seed: float = 96):
    """
    Imputes missing values using the MIDAS model.

    Parameters:
      df (pd.DataFrame): Input dataframe with NaNs in both numeric & categorical.
      continuous_cols (list): List of continuous column names.
      discrete_cols (list): List of discrete (numeric but not continuous) column names.
      categorical_cols (list): List of categorical column names.

    Returns:
      imps (list): A list of imputed dataframes, with original dtypes restored.
      method_info (str): Summary of MIDAS params used.
    """
    # 1. One-hot encode the categoricals
    if categorical_cols:
        md_cat_data, md_cats = md.cat_conv(df[categorical_cols])
    else:
        md_cat_data = pd.DataFrame(index=df.index)
        md_cats = []

    # 2. Build the “wide” DF: drop raw cats, append one-hots
    df_num = df.drop(columns=categorical_cols) if categorical_cols else df.copy()
    data_in = pd.concat([df_num, md_cat_data], axis=1)

    # 3. Record & re-insert the NaN locations so MIDAS sees them as missing
    na_mask = data_in.isnull()
    data_in[na_mask] = np.nan

    # 4. Scale only the numeric columns in place
    num_cols = discrete_cols + continuous_cols if (discrete_cols and continuous_cols) else []
    if num_cols:
        scaler = MinMaxScaler()
        data_in[num_cols] = scaler.fit_transform(data_in[num_cols])

    # 5. Build & train the MIDAS model
    imputer = md.Midas(
        layer_structure=layer,
        vae_layer=vae,
        seed=random_seed,
        input_drop=input_drop
    )
    imputer.build_model(data_in, softmax_columns=md_cats)
    imputer.train_model(training_epochs=training_epochs)

    # 6. Generate multiple imputations
    raw_imps = imputer.generate_samples(m=samples).output_list

    # 7. Decode each imputed DF back to original structure
    flat_cats = [c for grp in md_cats for c in grp]
    imps = []

    for imp_df in raw_imps:
        # 7a. inverse-scale numeric cols (if any)
        if num_cols:
            imp_df[num_cols] = scaler.inverse_transform(imp_df[num_cols])

        # 7b. decode one-hots (before dropping them!)
        decoded = {}
        for i, grp in enumerate(md_cats):
            present = [c for c in grp if c in imp_df.columns]
            decoded[categorical_cols[i]] = imp_df[present].idxmax(axis=1) if present else pd.Series([np.nan]*len(imp_df), index=imp_df.index)

        cat_df = pd.DataFrame(decoded, index=imp_df.index)

        # 7c. now drop the dummy cols
        base = imp_df.drop(columns=flat_cats, errors='ignore')

        # 7d. concat in your decoded cat columns
        merged = pd.concat([base, cat_df], axis=1)

        # 7e. round discrete cols
        if discrete_cols:
            merged[discrete_cols] = merged[discrete_cols].round().astype(int)

        imps.append(merged)

    method_info = f"MIDAS, params: samples={samples}, layer={layer}, vae={vae}, input_drop={input_drop}, epochs={training_epochs}"
    return imps, method_info

def select_best_imputations(imputed_dfs, original_df, mask_df, continuous_cols, discrete_cols, categorical_cols, method_info=None, method_names=None):
    """
    Evaluate one or several imputed DataFrames and determine an aggregated error.

    For each column with simulated missing data (per mask_df), numeric columns
    are scored using Mean Absolute Error (MAE) while categorical columns are scored
    by misclassification rate (1 - accuracy). An overall aggregated error is returned,
    which is the mean error over all evaluated columns.

    Parameters:
      imputed_dfs (list of pd.DataFrame): A list of imputed DataFrames.
      original_df (pd.DataFrame): The original (complete) DataFrame.
      mask_df (pd.DataFrame): Boolean DataFrame with True at positions where values are masked.
      continuous_cols (list): List of continuous numeric column names.
      discrete_cols (list): List of discrete numeric column names.
      categorical_cols (list): List of categorical column names.
      method_info (str, optional): Text description of the method and its hyperparameters.
      method_names (list, optional): List of names for each imputation method candidate.

    Returns:
      best_imputed_df (pd.DataFrame): A DataFrame where, for each column with missing values,
                                     the candidate with the lowest error is chosen.
      summary_table (pd.DataFrame): A summary table with metrics for each column.
      aggregated_error (float): The average error across columns (lower is better).
    """
    n_methods = len(imputed_dfs)
    
    if method_info is not None:
        parts = method_info.split(',')
        base_name = parts[0].strip()
        params = ','.join(parts[1:]).strip() if len(parts) > 1 else ""
        method_names = [f"{base_name} ({params})"] * n_methods
    elif method_names is None:
        method_names = [f"Method {i+1}" for i in range(n_methods)]
    
    summary_list = []
    best_method_per_col = {}

    for col in original_df.columns:
        if col in continuous_cols:
            col_type = "Continuous"
        elif col in discrete_cols:
            col_type = "Discrete"
        elif col in categorical_cols:
            col_type = "Categorical"
        else:
            col_type = str(original_df[col].dtype)

        if mask_df[col].sum() == 0:
            best_method_per_col[col] = None
            summary_list.append({
                'Column': col,
                'Data Type': col_type,
                'Best Method': None,
                'Metric': np.nan,  
            })
            continue

        col_errors = []
        for df_imp in imputed_dfs:
            if col_type in ["Continuous", "Discrete"]:
                try:
                    imp_vals = pd.to_numeric(df_imp[col][mask_df[col]], errors='coerce')
                    orig_vals = pd.to_numeric(original_df[col][mask_df[col]], errors='coerce')
                except Exception as e:
                    imp_vals = df_imp[col][mask_df[col]]
                    orig_vals = original_df[col][mask_df[col]]
                errors = np.abs(imp_vals - orig_vals)
                mae = errors.mean()
                col_errors.append(mae)
            else:
                correct = (df_imp[col][mask_df[col]] == original_df[col][mask_df[col]])
                accuracy = correct.mean()
                col_errors.append(1 - accuracy)

        if col_type in ["Continuous", "Discrete"]:
            best_idx = int(np.nanargmin(col_errors))
        else:
            best_idx = int(np.nanargmin(col_errors))
        best_method = method_names[best_idx]
        best_metric = col_errors[best_idx]

        best_method_per_col[col] = best_idx
        summary_list.append({
            'Column': col,
            'Data Type': col_type,
            'Best Method': best_method,
            'Metric': best_metric,
        })

    summary_table = pd.DataFrame(summary_list)
    
    best_imputed_df = original_df.copy()
    for cat in categorical_cols:
        if cat in best_imputed_df:
            best_imputed_df[cat] = best_imputed_df[cat].astype(object)

    for col in original_df.columns:
        if mask_df[col].sum() > 0 and best_method_per_col[col] is not None:
            method_idx = best_method_per_col[col]
            best_imputed_df.loc[mask_df[col], col] = \
                imputed_dfs[method_idx].loc[mask_df[col], col]

    errors = summary_table['Metric'].dropna().values
    aggregated_error = np.mean(errors) if len(errors) > 0 else np.nan

    return best_imputed_df, summary_table, aggregated_error

def optimize_imputation_hyperparams(imputation_func, 
                                    original_df, 
                                    df_missing, 
                                    mask_df, 
                                    continuous_cols, 
                                    discrete_cols, 
                                    categorical_cols, 
                                    timelimit=600,    # in seconds
                                    min_trials=20,
                                    random_seed=96):
    """
    Optimize hyperparameters for an imputation function using Optuna.

    This function takes the complete (original) DataFrame and a missing percentage.
    It uses `create_missings` to generate a DataFrame with simulated missing values and
    a corresponding mask. Then it runs the candidate imputation method on the incomplete
    DataFrame, evaluates the imputed results against the original DataFrame using the mask,
    and guides the hyperparameter search based on an aggregated error (lower is better).

    Parameters:
        imputation_func (callable): An imputation function (do_knn, do_mice, do_mf, or do_midas).
        original_df (pd.DataFrame): The complete ground-truth DataFrame.
        missing_percent (float): Percentage of missing values to simulate.
        continuous_cols (list): List of continuous numeric column names.
        discrete_cols (list): List of discrete numeric column names.
        categorical_cols (list): List of categorical column names.
        timelimit (int): Maximum time in seconds to run the optimization.
        min_trials (int): Minimum number of Optuna trials to run.
        random_seed (int): Seed for generating missingness (passed to create_missings).

    Returns:
        best_trial: The best trial object from the study.
        best_value: The best (lowest) aggregated objective value.
    """
    def objective(trial):
        func_name = imputation_func.__name__
        params = {}

        try:
            if func_name == "do_knn":
                # broadened search
                params['n_neighbors'] = trial.suggest_int("n_neighbors", 1, 20)
                params['scale'] = trial.suggest_categorical("scale", [True, False])
                imputed_df = imputation_func(df_missing, 
                                             continuous_cols=continuous_cols, 
                                             discrete_cols=discrete_cols, 
                                             categorical_cols=categorical_cols, 
                                             **params)
                imputed_dfs = [imputed_df]
                method_info = f"KNN, n_neighbors={params['n_neighbors']}, scale={params['scale']}"

            elif func_name == "do_mice":
                # more flexible ranges and include possibility of lower iterations
                params['iters'] = trial.suggest_int("iters", 3, 20)
                params['strat'] = trial.suggest_categorical("strat", ['normal', 'fast'])
                params['scale'] = trial.suggest_categorical("scale", [True, False])
                imputed_df = imputation_func(df_missing,
                                             continuous_cols=continuous_cols, 
                                             discrete_cols=discrete_cols, 
                                             categorical_cols=categorical_cols,
                                             **params)
                imputed_dfs = [imputed_df]
                method_info = f"MICE, iters={params['iters']}, strat={params['strat']}, scale={params['scale']}"

            elif func_name == "do_mf":
                params['iters'] = trial.suggest_int("iters", 1, 10)
                params['scale'] = trial.suggest_categorical("scale", [True, False])
                imputed_df = imputation_func(df_missing,
                                             continuous_cols=continuous_cols, 
                                             discrete_cols=discrete_cols, 
                                             categorical_cols=categorical_cols,
                                             **params)
                imputed_dfs = [imputed_df]
                method_info = f"MissForest, iters={params['iters']}, scale={params['scale']}"

            elif func_name == "do_midas":
                # improved search space
                params['num_layers'] = trial.suggest_int("num_layers", 1, 4)
                layers = []
                for i in range(params['num_layers']):
                    params[f"midas_layer_{i}"] = trial.suggest_categorical(f"midas_layer_{i}", [32, 64, 128, 256, 512])
                    layers.append(params[f"midas_layer_{i}"])
                params['layer'] = layers

                # tune whether vae layer is present
                params['vae'] = trial.suggest_categorical("vae", [True, False])

                # tune input_drop (how aggressively inputs are dropped during training)
                params['input_drop'] = trial.suggest_float("input_drop", 0.1, 0.9, step=0.05)

                # tune number of training epochs (smaller for quick search, larger for final build)
                params['training_epochs'] = trial.suggest_int("training_epochs", 5, 60)

                # number of samples returned by generate_samples
                params['samples'] = trial.suggest_int("samples", 3, 30)

                # call do_midas with only the relevant params
                call_kwargs = {
                    "layer": params['layer'],
                    "vae": params['vae'],
                    "samples": params['samples'],
                    "input_drop": params['input_drop'],
                    "training_epochs": params['training_epochs'],
                    "random_seed": random_seed,
                    "continuous_cols": continuous_cols,
                    "discrete_cols": discrete_cols,
                    "categorical_cols": categorical_cols
                }

                result = imputation_func(df_missing, **call_kwargs)

                if isinstance(result, tuple) and len(result) == 2:
                    imps, method_info = result
                else:
                    method_info = f"MIDAS, layer={layers}, vae={params['vae']}, samples={params['samples']}"
                    if isinstance(result, list):
                        imps = result
                    else:
                        imps = [result]

                imputed_dfs = imps if isinstance(imps, list) else [imps]

            else:
                raise ValueError(f"Unsupported imputation function: {func_name}")

            # Compute aggregated error
            _, _, aggregated_error = select_best_imputations(
                imputed_dfs, original_df, mask_df, continuous_cols, discrete_cols, categorical_cols,
                method_info=method_info
            )

            if np.isnan(aggregated_error):
                aggregated_error = 1e6

            return aggregated_error

        except Exception as e:
            # If any error happens inside a trial, return a large error so Optuna avoids it
            print(f"Trial failed for {func_name} with error: {e}")
            return 1e6

    # Use a seeded sampler for reproducibility
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, timeout=timelimit, n_trials=min_trials)

    best_trial = study.best_trial
    best_value = best_trial.value

    # Post-process best_trial.params so caller gets a clean dict (especially for MIDAS)
    raw_params = dict(best_trial.params)
    processed_params = dict(raw_params)  # shallow copy

    # If MIDAS was tuned, reconstruct 'layer' list from midas_layer_i params
    midas_layer_keys = sorted([k for k in raw_params.keys() if k.startswith("midas_layer_")],
                              key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else s)
    if midas_layer_keys:
        layers = [raw_params[k] for k in midas_layer_keys]
        processed_params['layer'] = layers
        # remove the midas_layer_i keys (optional)
        for k in midas_layer_keys:
            processed_params.pop(k, None)
        # ensure num_layers is present
        if 'num_layers' in raw_params:
            processed_params['num_layers'] = raw_params['num_layers']
        else:
            processed_params['num_layers'] = len(layers)

    print("Optimization completed!")
    print("Best Trial Hyperparameters:")
    for key, value in processed_params.items():
        print(f"  {key}: {value}")
    print(f"Best Objective Value (aggregated error): {best_value}")

    return best_trial, best_value, processed_params

def run_full_pipeline(df: pd.DataFrame, 
                      simulate: bool = False,
                      build: bool = False,
                      missingness_value: float = 10.0,
                      show_missingness: bool = False,
                      timelimit: int = 600,
                      min_trials: int = 20,
                      random_seed: int = 96):
    """
    Run a complete missing data imputation pipeline with method selection, 
    hyperparameter optimization, and optional final imputation build.

    This function simulates or detects missingness in the provided DataFrame, 
    optimizes hyperparameters for multiple imputation methods (KNN, MICE, MissForest, MIDAS),
    selects the best-performing method per column, and optionally builds a final
    imputed DataFrame using those best methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset. Should ideally have no missing values if `simulate=True`.

    simulate : bool, default False
        If True, simulates missingness in the dataset for evaluation purposes.
        If False, assumes the dataset already has missing values to impute.

    build : bool, default False
        If True and `simulate=True`, applies the best method per column 
        (based on the simulated evaluation) to impute the original input DataFrame.

    missingness_value : float, default 10.0
        Percentage of missing values to introduce in each column 
        when `simulate=False` and missingness is artificially created.

    show_missingness : bool, default False
        If True, shows a visualization of the missingness pattern 
        when simulating missing data.

    timelimit : int, default 600
        Time limit (in seconds) for hyperparameter tuning for each imputation method.

    min_trials : int, default 20
        Minimum number of trials during optimization for each method.

    random_seed : int, default 96
        Random seed for reproducibility of missingness and optimization.

    Returns
    -------
    best_imputed_df : pandas.DataFrame
        A DataFrame where missing values have been imputed using 
        the best-performing method per column.

    summary_table : pandas.DataFrame
        A summary of the best method selected per column including:
        - Column: Name of the column
        - Data Type: Continuous, Discrete, or Categorical
        - Best Method: Chosen imputation method
        - Metric: MAE (for numerical) or Accuracy (for categorical)
        - Error_SD: Standard deviation of errors
        - Max_Error: Maximum error observed
        - Min_Error: Minimum error observed
        - Within_10pct: Proportion of imputed values within 10% of true values

    Notes
    -----
    - This pipeline supports four imputation methods: 
      'KNN', 'MICE', 'MissForest', and 'MIDAS'.
    - If both `simulate=True` and `build=True`, the final output (`best_imputed_df`)
      is based on imputing the **original input `df`** using the optimal hyperparameters 
      derived from simulation.
    - MIDAS may return multiple imputed datasets; each is treated as a separate candidate.

    Examples
    --------
    >>> import pandas as pd
    >>> from mattlib.imputation import run_full_pipeline
    >>> df = pd.read_csv("clean_data.csv")
    >>> imputed_df, summary = run_full_pipeline(df, simulate=True, build=True)

    >>> # View best imputation methods per column
    >>> summary[['Column', 'Best Method', 'Metric']]

    >>> # Use the pipeline without building the final imputation
    >>> _, summary_only = run_full_pipeline(df, simulate=True, build=False)

    >>> # Impute an already incomplete dataset without simulation
    >>> df_missing = pd.read_csv("real_world_incomplete.csv")
    >>> imputed_df, summary = run_full_pipeline(df_missing, simulate=False)
    """
    # Suppress warnings for the entire pipeline run
    warnings.filterwarnings("ignore")

    try:
        if simulate:
            df_complete, df_missing, mask_df = simulate_missingness(
                df, show_missingness=show_missingness, random_state=random_seed
            )
        else:
            df_complete, df_missing, mask_df = create_missings(
                df, missingness=missingness_value, random_seed=random_seed
            )

        continuous_cols, discrete_cols, categorical_cols = prep(df)

        candidate_methods = {
            "KNN": do_knn,
            "MICE": do_mice,
            "MissForest": do_mf,
            "MIDAS": do_midas
        }

        best_hyperparams = {}

        for method_name, imputation_func in candidate_methods.items():
            print(f"\nOptimizing hyperparameters for {method_name}...")
            try:
                best_trial, best_value, best_params = optimize_imputation_hyperparams(
                    imputation_func=imputation_func,
                    original_df=df_complete,
                    df_missing=df_missing,
                    mask_df=mask_df,
                    continuous_cols=continuous_cols,
                    discrete_cols=discrete_cols,
                    categorical_cols=categorical_cols,
                    timelimit=timelimit,
                    min_trials=min_trials,
                    random_seed=random_seed
                )
                # store processed params so that downstream calls (especially MIDAS) get a proper 'layer' list
                best_hyperparams[method_name] = best_params
                print(f'Best hyperparameters for {method_name}: {best_hyperparams[method_name]} with best agg error of {best_value}')
            except Exception as e:
                print(f"An error occurred while optimizing {method_name}: {e}")
                best_hyperparams[method_name] = None

        imputed_dfs = []
        method_names = []

        for method in ['KNN', 'MICE', 'MissForest', 'MIDAS']:
            val = best_hyperparams.get(method)
            if not val:
                continue
            try:
                if method == 'KNN':
                    df_knn = do_knn(df_missing, continuous_cols=continuous_cols, 
                                    discrete_cols=discrete_cols, categorical_cols=categorical_cols, 
                                    n_neighbors=val.get('n_neighbors', 5), scale=val.get('scale', False))
                    imputed_dfs.append(df_knn)
                    method_names.append('KNN')

                elif method == 'MICE':
                    df_mice = do_mice(df_missing, continuous_cols=continuous_cols, 
                                      discrete_cols=discrete_cols, categorical_cols=categorical_cols, 
                                      iters=val.get('iters', 10), strat=val.get('strat', 'normal'), scale=val.get('scale', False))
                    imputed_dfs.append(df_mice)
                    method_names.append('MICE')

                elif method == 'MissForest':
                    df_mf = do_mf(df_missing, continuous_cols=continuous_cols, 
                                  discrete_cols=discrete_cols, categorical_cols=categorical_cols, 
                                  iters=val.get('iters', 5), scale=val.get('scale', False))
                    imputed_dfs.append(df_mf)
                    method_names.append('MissForest')

                elif method == 'MIDAS':
                    # ensure required defaults
                    mids_layer = val.get('layer', [256, 256])
                    mids_vae = val.get('vae', True)
                    mids_samples = val.get('samples', 10)
                    mids_input_drop = val.get('input_drop', 0.75)
                    mids_epochs = val.get('training_epochs', 20)

                    df_midas_list, _ = do_midas(df_missing, continuous_cols=continuous_cols,
                                                discrete_cols=discrete_cols,
                                                categorical_cols=categorical_cols,
                                                layer=mids_layer, vae=mids_vae, 
                                                samples=mids_samples,
                                                input_drop=mids_input_drop,
                                                training_epochs=mids_epochs,
                                                random_seed=random_seed)
                    imputed_dfs.extend(df_midas_list)
                    method_names.extend([f'MIDAS_{i+1}' for i in range(len(df_midas_list))])

            except Exception as e:
                print(f"Failed to impute with {method}: {e}")

        best_method_per_col = {}
        summary_list = []

        for col in df_missing.columns:
            if col in continuous_cols:
                col_data_type = "Continuous"
            elif col in discrete_cols:
                col_data_type = "Discrete"
            elif col in categorical_cols:
                col_data_type = "Categorical"
            else:
                col_data_type = str(df_missing[col].dtype)

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
                for df_imp in imputed_dfs:
                    imp_vals = pd.to_numeric(df_imp[col][mask_df[col]], errors='coerce')
                    orig_vals = pd.to_numeric(df_complete[col][mask_df[col]], errors='coerce')
                    errors = np.abs(imp_vals - orig_vals)
                    mae = errors.mean() if not errors.empty else np.nan
                    metrics.append(mae)
                best_idx = np.nanargmin(metrics)
                best_metric = metrics[best_idx]

                best_imp_vals = pd.to_numeric(imputed_dfs[best_idx][col][mask_df[col]], errors='coerce')
                best_orig_vals = pd.to_numeric(df_complete[col][mask_df[col]], errors='coerce')
                errors = np.abs(best_imp_vals - best_orig_vals)
                error_sd = errors.std() if not errors.empty else np.nan
                max_error = errors.max() if not errors.empty else np.nan
                min_error = errors.min() if not errors.empty else np.nan
                condition = ((best_orig_vals != 0) & (errors <= 0.1 * best_orig_vals.abs())) | \
                            ((best_orig_vals == 0) & (errors == 0))
                within_10pct = condition.mean() if not condition.empty else np.nan

            elif col in categorical_cols or pd.api.types.is_string_dtype(df_complete[col]):
                for df_imp in imputed_dfs:
                    correct = (df_imp[col][mask_df[col]] == df_complete[col][mask_df[col]])
                    acc = correct.mean() if not correct.empty else np.nan
                    metrics.append(acc)
                best_idx = np.nanargmax(metrics)
                best_metric = metrics[best_idx]

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

        best_imputed_df = df_complete.copy()
        for col in df_complete.columns:
            if mask_df[col].sum() > 0 and best_method_per_col[col] is not None:
                method_idx = best_method_per_col[col]
                best_imputed_df.loc[mask_df[col], col] = imputed_dfs[method_idx].loc[mask_df[col], col]

        # Build the final imputed DataFrame if both build and simulate are True
        if build and simulate:
            if df.isnull().sum().sum() == 0:
                print("Original DataFrame has no missing values. Build is not needed.")
                final_imputed_df = df.copy()
            else:
                imputed_dfs_build = []
                method_names_build = []

                for method in ['KNN', 'MICE', 'MissForest', 'MIDAS']:
                    val = best_hyperparams.get(method)
                    if not val:
                        continue
                    try:
                        if method == 'KNN':
                            df_knn_build = do_knn(df, continuous_cols=continuous_cols, 
                                                discrete_cols=discrete_cols, categorical_cols=categorical_cols, 
                                                n_neighbors=val.get('n_neighbors', 5), scale=val.get('scale', False))
                            imputed_dfs_build.append(df_knn_build)
                            method_names_build.append('KNN')

                        elif method == 'MICE':
                            df_mice_build = do_mice(df, continuous_cols=continuous_cols, 
                                                  discrete_cols=discrete_cols, categorical_cols=categorical_cols, 
                                                  iters=val.get('iters', 10), strat=val.get('strat', 'normal'), scale=val.get('scale', False))
                            imputed_dfs_build.append(df_mice_build)
                            method_names_build.append('MICE')

                        elif method == 'MissForest':
                            df_mf_build = do_mf(df, continuous_cols=continuous_cols, 
                                              discrete_cols=discrete_cols, categorical_cols=categorical_cols, 
                                              iters=val.get('iters', 5), scale=val.get('scale', False))
                            imputed_dfs_build.append(df_mf_build)
                            method_names_build.append('MissForest')

                        elif method == 'MIDAS':
                            df_midas_list_build, _ = do_midas(df, continuous_cols=continuous_cols,
                                                            discrete_cols=discrete_cols,
                                                            categorical_cols=categorical_cols,
                                                            layer=val.get('layer', [256, 256]), vae=val.get('vae', True), 
                                                            samples=val.get('samples', 10),
                                                            input_drop=val.get('input_drop', 0.75),
                                                            training_epochs=val.get('training_epochs', 20),
                                                            random_seed=random_seed)
                            imputed_dfs_build.extend(df_midas_list_build)
                            method_names_build.extend([f'MIDAS_{i+1}' for i in range(len(df_midas_list_build))])

                    except Exception as e:
                        print(f"Failed to impute with {method} during build: {e}")

                # Map method names to their imputed DataFrames
                method_to_imp_df = dict(zip(method_names_build, imputed_dfs_build))
                final_imputed_df = df.copy()
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        col_summary = summary_table[summary_table['Column'] == col]
                        if col_summary.empty:
                            continue
                        best_method = col_summary['Best Method'].iloc[0]
                        if best_method and best_method in method_to_imp_df:
                            imp_df = method_to_imp_df[best_method]
                            missing_mask = df[col].isnull()
                            final_imputed_df.loc[missing_mask, col] = imp_df.loc[missing_mask, col]

                best_imputed_df = final_imputed_df

        return best_imputed_df, summary_table

    finally:
        # restore default warning behaviour after the pipeline finishes
        warnings.filterwarnings("default")
