import numpy as np
import logging
import sys
from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int, int2ba
import pandas as pd
from typing import Callable, Any
BIT_ORDER="little"
logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG )
logger = logging.getLogger()
sub_s = "₀₁₂₃₄₅₆₇₈₉"

# Function to encode the binary variables
def bin_encoder(v, l):
    """
    Encode a Boolean variable into an integer.

    Args:
        v (int): The Boolean variable to encode.
        l (int): The length of the binary representation.
    Returns:
        int: The encoded integer.
    """
    y = bitarray(l, endian=BIT_ORDER)
    y[v] = 1
    return ba2int(y)

# Function to create П matrix
def create_perm(C: np.array, sort_func=None):
    """
    Generate a permutation indicator matrix based on the column-wise sorting of a given numpy array.

    Parameters:
        C (numpy.ndarray): The input numpy array.
        sort_func (callable, optional): Sorting function to use. If None, uses default ascending sort.

    Returns:
        numpy.ndarray: The permutation indicator matrix.
    """
    if sort_func is None:
        # Default behavior - ascending sort
        perm = C.argsort(kind='quick', axis=0)
    else:
        # Use provided sorting function
        perm = sort_func(C)
    return perm

def calculate_degree(y):
    """
    Calculate the degree of a Boolean variable.

    Args:
        y (int): The Boolean variable to calculate the degree of.

    Returns:
        int: The degree of the Boolean variable.
    """
    y_bin = int2ba(y, endian=BIT_ORDER)
    degree = y_bin.count()
    return degree

# Function to create the coefficients using the П matrix
def create_coeffs_matrix(C: np.array, perm: np.array):
    """
    Create a coefficients matrix based on the given matrix C and permutation indicator matrix perm.

    Args:
        C (numpy.ndarray): The input matrix C.
        perm (numpy.ndarray): The permutation indicator matrix.

    Returns:
        numpy.ndarray: The coefficients matrix.
    """
    sorted_c = np.take_along_axis(C, perm, axis=0)
    zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)
    coeffs_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))
    coeffs_c = coeffs_c[:-1]
    return coeffs_c

# Function to create the Boolean variables
def create_variable_matrix(C: np.array, perm: np.array):
    """
    Create a variable matrix based on the given matrix C and permutation indicator matrix perm.

    Args:
        C (numpy.ndarray): The input matrix C.
        perm (numpy.ndarray): The permutation indicator matrix.

    Returns:
        numpy.ndarray: The variable matrix.
    """
    y = perm[:-1]
    y_maker = np.frompyfunc(bin_encoder, 2, 1)
    y = y_maker(y, perm.shape[0])
    # print(y)
    y = y.cumsum(axis=0)
    return y

# Function to reduce the polynomial
def reduce_pbp_pandas(coeffs: np.array, variables: np.array, agg_func: Callable[[pd.Series], Any] = lambda x: x.sum()):
    """
    Reduces a polynomial basis representation using pandas DataFrame operations.

    Args:
        coeffs (numpy.ndarray): An array of coefficients representing the polynomial basis.
        variables (numpy.ndarray): An array of variables representing the polynomial basis.

    Returns:
        pandas.DataFrame: A DataFrame containing the reduced polynomial basis representation.
    """
    zero_vars = np.zeros((1, variables.shape[1]), dtype=int)
    var_flat = np.vstack([zero_vars, variables]).ravel()
    df = pd.DataFrame()
    df["y"] = var_flat
    df["coeffs"] = coeffs.ravel()
    df = df.groupby(['y'], as_index=False).agg({'y': 'first', 'coeffs': agg_func })
    zero_coeffs = df["coeffs"] == 0
    df = df.loc[~zero_coeffs]
    df["degree"] = df["y"].apply(calculate_degree)
    df.sort_values(by=['degree'], inplace=True)
    blankIndex=[''] * len(df)
    df.index=blankIndex
    return df

# Function to decode Boolean variables
def decode_var(y, BIT_ORDER=BIT_ORDER):
    """
    Decode a variable from its binary representation.

    Parameters:
        y (int): The binary representation of the variable.

    Returns:
        str: The decoded variable as a string.
    """
    bin_indices = int2ba(y, endian=BIT_ORDER)
    y_arr  = np.frombuffer(bin_indices.unpack(), dtype=bool)
    indices = np.nonzero(y_arr)[0]
    if indices.size == 0:
        return ""
    
    return "y" + "y".join([sub_s[i+1] for i in indices])

# Driver function to create a whole pBp
def create_pbp(c: np.array, agg_func: Callable[[pd.Series], Any] = lambda x: x.sum(), sort_func=None):
    """
    Create a polynomial basis representation (pBp) based on the given matrix C.

    Args:
        c (numpy.ndarray): The input matrix C.
        agg_func (callable, optional): Aggregation function to use. Defaults to sum.
        sort_func (callable, optional): Sorting function to use. Defaults to ascending sort.

    Returns:
        pandas.DataFrame: A DataFrame containing the polynomial basis representation.
    """
    assert len(c.shape) == 2
    perm_c = create_perm(c, sort_func)
    coeffs_c = create_coeffs_matrix(c, perm_c)
    y = create_variable_matrix(c, perm_c)
    pBp = reduce_pbp_pandas(coeffs_c, y, agg_func)
    return pBp

# Function to truncate a pBp by a given p value
def truncate_pBp(pBp, c, p):
    """
    Truncates the polynomial basis representation (pBp) based on the given parameters.

    Parameters:
        pBp (pandas.DataFrame): The original polynomial basis representation.
        c (numpy.array): The input matrix C used for truncation.
        p (int): The cutoff value for truncation.

    Returns:
        pandas.DataFrame: The truncated polynomial basis representation.
    """
    cutoff = c.shape[0] - p + 1
    truncated_pBp = pBp.loc[pBp['degree'] < cutoff]
    return truncated_pBp

# Function to show terms added together as a polynomial
def to_string(row):
    """
    Returns a string representation of a row in a DataFrame.

    Args:
        row (pandas.Series): A row from a DataFrame.

    Returns:
        str: A string representation of the row.
    """
    return f'{row["coeffs"] if row["coeffs"] > 1 else "" }{row["y_str"]}'

# Driver function to create and truncate pBp
def trunc_driver(c, p_list, agg_func=None, sort_func=None):
    """
    Truncates the polynomial basis representation (pBp) for a given input matrix C and a list of cutoff values p.

    Parameters:
        c (numpy.array): The input matrix C.
        p_list (list): A list of cutoff values p.
        agg_func (callable, optional): Aggregation function to use. Defaults to sum.
        sort_func (callable, optional): Sorting function to use. Defaults to ascending sort.

    Returns:
        None
    """
    pBp = create_pbp(c, agg_func, sort_func)
    print("Result pBp")
    polynomial = " + ".join(pBp.apply(to_string, axis=1))
    print(polynomial)
    print("=" * 100)
    for p in p_list:
        truncated_pBp = truncate_pBp(pBp, c, p)
        polynomial = " + ".join(truncated_pBp.apply(to_string, axis=1))
        print(f"p = {p}")
        print(polynomial)
        print("=" * 100)


def pbp_vector(c: np.array, agg_func: Callable[[pd.Series], Any] = lambda x: x.sum(), sort_func=None):
    """
    Creates a polynomial basis representation (pBp) for a given input matrix C.
    
    Args:
        c (numpy.ndarray): The input matrix C.
        agg_func (callable, optional): Aggregation function to use. Defaults to sum.
        sort_func (callable, optional): Sorting function to use. Defaults to ascending sort.
    
    Returns:
        numpy.ndarray: The PBP vector representation.
    """
    pBp = create_pbp(c, agg_func, sort_func)
    vector = np.zeros(2**c.shape[0] - 1, dtype=c.dtype)
    for index, row in pBp.iterrows():
        vector[row["y"]] = row["coeffs"]
    
    return vector


def get_pbp_from_vector(vector: np.array):
    """
    Creates a polynomial basis representation (pBp) from a given vector.
    """
    occupied_positions = np.nonzero(vector)[0]
    y = np.zeros(vector.shape[0], dtype=int)
    
    pbp = pd.DataFrame()
    pbp["y"] = occupied_positions
    pbp["coeffs"] = vector[occupied_positions]
    pbp["degree"] = pbp["y"].apply(calculate_degree)
    pbp["y_str"] = pbp["y"].apply(lambda x: decode_var(x))
    pbp.sort_values(by=['degree'], inplace=True)
    blankIndex=[''] * len(pbp)
    pbp.index=blankIndex


    return pbp

if __name__ == "__main__":
    c = np.array([
        [7, 8, 2, 10, 3],
        [4, 12, 1, 8, 4],
        [5, 3, 0, 6, 9],
        [9, 6, 7, 1, 5]
    ])

    pBp = create_pbp(c)
    print(pBp)
    v = pbp_vector(c)
    print(v)
    print(get_pbp_from_vector(v))
    

    # c = np.array([
    #     [5.4, 3.4],
    #     [1.7, 0.2],
    # ])
    # v = pbp_vector(c)
    # print(v)

    c = np.array([
        [5.1, 3.7],
        [1.5, 0.4],
    ]) 