import os
import pandas as pd
import numpy as np
try:
    from numba import njit
except Exception:
    # Fallback when numba is unavailable/incompatible; keep functions usable.
    def njit(*jit_args, **jit_kwargs):
        if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
            return jit_args[0]

        def _decorator(func):
            return func

        return _decorator
import statsmodels.api as sm

if os.environ.get("USE_RPY2", "0") == "1":
    import rpy2.robjects as ro # type: ignore
    from rpy2.robjects.packages import importr # type: ignore
    from rpy2.robjects import pandas2ri # type: ignore
    from rpy2.robjects.conversion import localconverter # type: ignore
    try:
        pandas2ri.activate()
    except DeprecationWarning:
        pass
    utils = importr('utils')
    devtools = importr('devtools')
    linda = importr("LinDA")


def normalize(X):
    """
    transforms to the simplex X should be of a pd.DataFrame of form (p, N)
    """
    return X / X.sum(axis=0)


def geometric_mean(x, positive=False):
    """
    calculates the geometric mean of a vector
    """
    assert not np.all(x == 0)

    if positive:
        x = x[x > 0]
    a = np.log(x)
    g = np.exp(a.sum() / len(a))
    return g


def log_transform(X, transformation=str, eps=0.1):
    """
    log transform, scaled with geometric mean X should be a pd.DataFrame of form (p,N)
    """
    if transformation == "clr":
        assert not np.any(X.values == 0), "Add pseudo count before using clr"
        g = X.apply(geometric_mean)
        Z = np.log(X / g)
    elif transformation == "mclr":
        g = X.apply(geometric_mean, positive=True)
        X_pos = X[X > 0]
        Z = np.log(X_pos / g)
        Z = Z + abs(np.nanmin(Z.values)) + eps
        Z = Z.fillna(0)
    return Z


def calculate_correlation(counts, sample_ids):
    """
    Calculate empirical correlation network.

    Parameters:
    - counts (pd.DataFrame): The DataFrame containing bacteria counts.
    - sample_ids (list): list of str: List of sample IDs corresponding to the specified status.

    Returns:
    - corr (np.ndarray): The covariance matrix after processing.
    """
    counts_selected = counts[sample_ids]
    clr_counts = transform_features(counts_selected, transformation="clr")
    S = np.cov(clr_counts, bias=True)
    corr = scale_array_by_diagonal(S)

    return corr

def zero_imputation(df: pd.DataFrame, pseudo_count: int = 1):
    """
    Perform zero imputation on a DataFrame by replacing zero values with a pseudo count,
    and then scaling the columns to maintain the original sum of values.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric values.

    pseudo_count : int, optional
        The value used to replace zero values, defaults to 1.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with zero-imputed and scaled values.
    """
    X = df.copy()
    original_sum = X.sum(axis=0) # sum per row (sample)
    for col in X.columns:
        X[col].replace(to_replace=0, value=pseudo_count, inplace=True)
    shifted_sum = X.sum(axis=0) # sum per row (sample)
    scaling_parameter = original_sum.div(shifted_sum)
    X = X.mul(scaling_parameter, axis =1) # multiply by column

    return X


def r_to_pandas(r_dataframe):
    """
    Convert an R DataFrame to a Pandas DataFrame.

    Parameters:
    - r_dataframe: An R DataFrame.

    Returns:
    pd.DataFrame: A Pandas DataFrame equivalent to the input R DataFrame.
    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.rpy2py(r_dataframe)
        
    return df



def transform_features(X: pd.DataFrame, transformation: str = "clr", pseudo_count: int = 1) -> pd.DataFrame:
    """
    Project compositional data to Euclidean space.

    Parameters
    ----------
    pseudo_count: int, optional
        Add pseudo count, only necessary for transformation = "clr".
    table: biom.Table
        A table with count microbiome data.
    transformation: str
        If 'clr' the data is transformed with center log-ratio method by Aitchison (1982).
        If 'mclr' the data is transformed with modified center log-ratio method by Yoon et al. (2019).

    Returns
    -------
    X: pd.Dataframe
        Count data projected to Euclidean space.

    """
    columns = X.columns

    if transformation == "clr":
        X = zero_imputation(X, pseudo_count=pseudo_count)
        X = normalize(X)
        X = log_transform(X, transformation=transformation)

        return pd.DataFrame(X, columns=columns)

    elif transformation == "mclr":
        X = normalize(X)
        X = log_transform(X, transformation=transformation)

        return pd.DataFrame(X, columns=columns)

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


def read_data_from_csv(file_path, sep: str=";", encoding: str='unicode_escape'):
    """
    Read data from a CSV file using pandas.

    This function reads data from a CSV file located at the specified `file_path`
    and returns a pandas DataFrame with the contents of the CSV.

    Parameters:
    file_path (str): The path to the CSV file to be read.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        data_df = pd.read_csv(file_path, on_bad_lines='skip', encoding=encoding, sep=sep)
        return data_df
    except FileNotFoundError:
        raise FileNotFoundError("The specified CSV file does not exist.")
    except pd.errors.ParserError:
        raise pd.errors.ParserError("Failed to parse the CSV file. Check the file format and content.")
        
        
def calculate_log_ratios(raw_counts, pseudo = int):
    """
    Calculate pairwise log-ratios for each sample without transposing the dataframe.
    
    Parameters:
    -----------
    raw_counts: pd.DataFrame
        A DataFrame where each row represents a taxon and each column represents a sample. 
        The DataFrame should contain non-negative counts of taxa across samples.
        
    pseudo: int, optional, default=1
        A pseudo-count to add to each count in the DataFrame to avoid taking the logarithm of zero.
        This prevents errors from zero counts and stabilizes variance for low counts.

    Returns:
    --------
    log_ratios: pd.DataFrame
        DataFrame with log-ratios where each column corresponds to a pairwise
        log-ratio of taxa for each sample.
    """

    raw_counts = raw_counts.applymap(lambda x: x + pseudo if x == 0 else x)
    
    n_taxa, n_samples = raw_counts.shape
    
    # Create indices for the upper triangular matrix (excluding the diagonal)
    idx0, idx1 = np.triu_indices(n_taxa, 1)
    
    # Collect log-ratios in a list
    log_ratio_list = []

    # Calculate log-ratios for each pair of taxa
    for row_idx, col_idx in zip(idx0, idx1):
        taxa1, taxa2 = raw_counts.index[row_idx], raw_counts.index[col_idx]
        
        log_ratio = np.log10(raw_counts.iloc[row_idx].values) - np.log10(raw_counts.iloc[col_idx].values)
        
        log_ratio_list.append(pd.DataFrame({f'{taxa1}_vs_{taxa2}': log_ratio}, index=raw_counts.columns))

    log_ratios = pd.concat(log_ratio_list, axis=1)
    
    return log_ratios.T
        
        
def scale_array_by_diagonal(X, d=None):
    """
    Scales a 2D-array X by the diagonal matrix W, where W^2 = diag(d).

    Parameters:
    X (numpy.ndarray): The input 2D-array to be scaled.
    d (numpy.ndarray or None, optional): Diagonal elements for the scaling matrix W.
        If None, the diagonal elements are set to the square root of the diagonal of X.
    
    Returns:
    numpy.ndarray: The scaled array, computed as X_ij / sqrt(d_i * d_j).

    Notes:
    The scaling operation can be expressed as W^-1 @ X @ W^-1, where W^2 = diag(d).
    If d is not provided, the diagonal elements of W are computed as the square root of
    the diagonal elements of X.

    Reference:
    Equation (2.4) from "Covariance Matrix Selection and Estimation via Penalized Normal
    Likelihood" by Olivier Ledoit and Michael Wolf.
    Link: https://fan.princeton.edu/papers/09/Covariance.pdf

    Raises:
    AssertionError: If X is not a 2D array or if the length of d doesn't match the number of rows in X.
    """
    assert len(X.shape) == 2
    if d is None:
        d = np.diag(X)
    else:
        assert len(d) == X.shape[0]

    scale = np.tile(np.sqrt(d), (X.shape[0], 1))
    scale = scale.T * scale

    return X / scale
        

def save_dataframe(data, file_path, index = False):
    """
    data: pd.Dataframe
    filename: str
    """
    data.to_csv(file_path, index=index)
    return data


def move_column_to_front(dataframe, column_name):
    """
    Move a specified column to the front of a pandas DataFrame.

    This function takes a pandas DataFrame and the name of the column to move, 
    and it moves that column to the front of the DataFrame. The function modifies
    the DataFrame in place and does not return anything.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column to move to the front.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    id_column = dataframe.pop(column_name)
    dataframe.insert(0, column_name, id_column)
    return dataframe


def _count_missing(df):
    """
    Counts the share of missing values in each column of a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        None
    """
    stats = df.isna().mean(axis=0)

    stats_df = pd.DataFrame(stats, columns=["Share of missing values"])
    stats_df = stats_df.sort_values(by="Share of missing values", ascending=False)
    return stats_df


def _flatten_list(d=dict):
    """
    Flattens a dictionary of lists into a single list.

    Args:
        d (dict): The input dictionary.

    Returns:
        list: The flattened list.
    """
    t = list(d.values())
    flat_list = [item for sublist in t for item in sublist]
    
    return flat_list


def check_samples_overlap(df, asv, show_diff=True):
    """
    Filter and report samples in covariates DataFrame 'df' based on their presence in count data DataFrame 'asv'.

    This function takes two DataFrames, 'df' and 'asv', and performs the following operations:
    1. Converts the index of 'df' to a list of strings.
    2. Finds the common samples between the index of 'df' and the columns of 'asv'.
    3. Reports the number of samples present in 'df' but not in 'asv' if any.
    4. Excludes samples from 'df' that are not present in 'asv'.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing samples.
    - asv (pandas.DataFrame): The DataFrame used for comparison.
    - show_diff (bool): Whether to print the samples present in 'df' but not in 'asv'. Default is True.

    Returns:
    - pandas.DataFrame: The filtered DataFrame after excluding samples not present in 'asv'.
    """
    str_id = list(map(str, df.index))
    common_samples = list(set(str_id).intersection(set(asv.columns)))
    diff_samples = list(set(str_id).difference(set(asv.columns)))
    if diff_samples and show_diff:
        print(f"Samples that are present in matched pairs, but not in ASVs: {len(diff_samples)}")
        for sample in diff_samples:
            print(sample)

    # Exclude samples which are not present in ASVs (if any)
    int_id = list(map(int, common_samples))
    df = df[df.index.isin(int_id)]
    return df, diff_samples



def filter_by_column(ige, column_name):
    """
    Filters rows in a DataFrame where a specific column has a value of 1, and all other columns are 0.

    Parameters:
    ige (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to filter by.

    Returns:
    pd.DataFrame: A DataFrame containing rows where the specified column is 1 and all other columns are 0.
    """
    filtered_ige = ige[(ige[column_name] == 1) & (ige.drop(column_name, axis=1).sum(axis=1) == 0)]
    
    return filtered_ige


def perform_bh_correction_and_filter(taxa, stats, alpha=0.05):
    """
    Perform Benjamini-Hochberg correction on p-values and filter significant features.

    Parameters:
    - taxa (DataFrame): DataFrame containing taxonomic information.
    - stats (DataFrame): DataFrame containing statistical results with 'p_value' column.
    - alpha (float, optional): Threshold for significance. Default is 0.05.

    Returns:
    - DataFrame: Subset of the input DataFrame containing features with adjusted p-values <= alpha.
    """
    # Perform Benjamini-Hochberg correction
    adjusted_pvalues = sm.stats.multipletests(stats['p_value'], method='fdr_bh')[1]

    # Update the DataFrame with adjusted p-values
    stats['adjusted_pvalue'] = adjusted_pvalues
    stats['effect'] = np.sign(stats['obs_stat'])

    # Display the result
    da_species = stats[stats['adjusted_pvalue'] <= alpha]

    return da_species



def calculate_obs_stat_and_pvalues(taxa_dict, w):
    """
    Calculate observed statistics and asymptotic p-values for each taxonomic level.

    Parameters:
    - taxa_dict (dict): Dictionary containing taxonomic levels as keys and DataFrames as values.
    - w (pd.DataFrame): DataFrame containing labels.

    Returns:
    - obs_stat_dict (dict): Dictionary containing taxonomic levels as keys and observed statistics as values.
    - asymptotic_pvalues (dict): Dictionary containing taxonomic levels as keys and asymptotic p-values as values.
    """
    result_dict = dict()

    for level in taxa_dict.keys():
        print(level)

        data = taxa_dict[level]

        p, N = data.shape

        print(p, N)
        print(w.shape)

        if p >= 20:
            lo = linda.linda(data, w, formula="~w", alpha=0.05, prev_cut=0, lib_cut=1)

            linda_out = r_to_pandas(lo.rx2("output"))['w']

            # observed statistic of true W            
            stats = pd.DataFrame({
            "obs_stat": linda_out['stat'],
            "p_value": linda_out['pvalue'],
            "reject_null": linda_out['reject'],
            'lfc': linda_out['log2FoldChange'],
            'lfcSE': linda_out['lfcSE'],
            'baseMean': linda_out['baseMean'] })
            
            result_dict[level] = stats

    return result_dict



def generate_taxa_dict(asv, taxa):
    """
    Generate a dictionary of count tables for each taxonomic level.

    Parameters:
    - asv (pd.DataFrame): DataFrame containing ASV samples.
    - taxa (pd.DataFrame): DataFrame containing taxonomic information.

    Returns:
    dict: A dictionary where keys are taxonomic levels, and values are DataFrames with count tables.
    """
    taxa_dict = dict()
    
    asv_with_taxon = asv.join(taxa['name'])

    for level in taxa.columns:
        if level != 'name':
            df_level = asv_with_taxon.join(taxa[level])
            name_dict = df_level.set_index(level)['name'].to_dict()
            df_level = df_level.groupby(level).sum()
            df_level.index = df_level.index.to_series().replace(name_dict)

            df_level.drop(columns=['name'], inplace=True)


        taxa_dict[level] = df_level

    taxa_dict["ASVs"] = asv

    return taxa_dict


def calculate_column_counts(df):
    """
    Calculates the total number of non-zero elements for each column in a DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with column counts.
    """
    column_counts = df.astype(bool).sum(axis=0)
    column_counts = pd.DataFrame(column_counts, columns=['counts'])
    
    return column_counts


def _lowercase(input_dict):
    """
    Convert the values in the input dictionary to lowercase.

    This function takes a dictionary as input and converts all the values to lowercase.
    If a value is a list, it converts each element of the list to lowercase.
    If a value is not a list, it directly converts it to lowercase.

    Parameters:
    input_dict (dict): The input dictionary to be processed.

    Returns:
    dict: A new dictionary with all values converted to lowercase.
    """
    for key in input_dict:
        if isinstance(input_dict[key], list):
            input_dict[key] = [value.lower() for value in input_dict[key]]
        else:
            input_dict[key] = input_dict[key].lower()
            
    return input_dict



def calculate_unadjusted_p_values(test_stats, obs_stat, test_type):
    """
    Calculate unadjusted p-values for a given test statistic and test type.

    Parameters:
        test_stats (pandas.DataFrame): A DataFrame containing test statistics.
        obs_stat (pandas.Series): The observed test statistic.
        test_type (str): The type of statistical test ("two-sided" or "one-sided").

    Returns:
        pandas.DataFrame: A DataFrame with unadjusted p-values for each observation.
    """

    if test_type == "two-sided":
        # two-sided test with absolute value tests both directions
        
        # calculate the proportion of |T_rand| >= |T_obs|
        p_value = test_stats.abs().ge(obs_stat.abs().values, axis = "rows")
        p_value.index = obs_stat.index

        p_value = p_value.mean(axis = "columns").sort_values().to_frame()
        p_value.columns = ["p-value"]
    
    return p_value



def min_unadjusted_p_values(test_stats):
    """
    Calculates the minimum unadjusted p-values for a given set of test statistics.

    Args:
        test_stats (DataFrame): A DataFrame of test statistics, where each column represents a different test statistic.

    Returns:
        np.array: An array of the minimum unadjusted p-values.
    """
    
    min_p_values = list()

    for i in test_stats.columns:
        
        # choose random statistic as "observed" statistic
        obs_stat = test_stats.loc[:, i]

        # select random statistics except the observed one
        t = test_stats.loc[:, test_stats.columns != i]

        # select statistics greater or equal to the observed one
        t_comp = t.abs().ge(obs_stat.abs().values, axis = "rows")

        # calculate unadjusted  p-values
        t_comp = t_comp.mean(axis = "columns")

        # save min p-value among all species
        min_p_values.append(min(t_comp))
        
    min_p_values = np.array(min_p_values)
    
    return min_p_values


def adjusted_p_values(min_p_values, p_value):
    """
    Calculates the adjusted p-values for a given set of minimum unadjusted p-values and p-values.

    Args:
        min_p_values (np.array): An array of minimum unadjusted p-values.
        p_value (DataFrame): A DataFrame of p-values, where each row represents a different test statistic and each column represents a different species.

    Returns:
        np.array: An array of adjusted p-values.
    """
    
    adj_p_values = list()

    # select p-values smaller or equal to unadjusted p-values
    p, _ = p_value.shape
    
    for i in range(0, p):
        adj = np.mean(min_p_values <= p_value['p-value'][i])
        adj_p_values.append(adj)
    
    return adj_p_values


@njit() 
def prox_od_1norm(A, l, diag=True):
    """
    calculates the prox of the off-diagonal 1norm at a point A

    diag : bool, optional (default=False)
    If True, preserves the diagonal elements of A.
    If False, applies soft-thresholding to all elements.

    """    
    (d1,d2) = A.shape
    res = np.sign(A) * np.maximum(np.abs(A) - l, 0.)

    if diag:
        for i in np.arange(np.minimum(d1,d2)):
            res[i,i] = A[i,i]
    
    return res