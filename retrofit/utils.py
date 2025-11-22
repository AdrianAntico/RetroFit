# Module: utils
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.7
# Last modified : 2021-09-21

import pickle
import os
import numpy as np
import polars as pl
from datetime import date, timedelta


def make_retrofit_demo_data(
    n_rows: int = 50_000,
    n_segments: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Build a synthetic dataset tailored for RetroFit demos:

    Columns
    -------
    CalendarDateColumn : date
    XREGS1, XREGS2, XREGS3 : float
    MarketingSegments, MarketingSegments2, MarketingSegments3 : str
    Leads : float (regression target)
    Label : str (multiclass target, also usable for binary if you collapse)
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1. Base structure
    # -----------------------------
    n = n_rows

    # Dates over ~2 years
    start_date = date(2022, 1, 1)
    dates = np.array([start_date + timedelta(days=int(d)) for d in rng.integers(0, 730, size=n)])

    # Segments
    seg_levels = [f"S{i}" for i in range(1, n_segments + 1)]
    seg = rng.choice(seg_levels, size=n, p=np.linspace(1.0, 2.0, n_segments) / np.linspace(1.0, 2.0, n_segments).sum())
    seg2 = rng.choice(seg_levels, size=n)
    seg3 = rng.choice(seg_levels, size=n)

    # Encode segments numerically for interactions
    seg_to_num = {s: i for i, s in enumerate(seg_levels, start=1)}
    seg_num = np.vectorize(seg_to_num.get)(seg)

    # -----------------------------
    # 2. Numeric drivers (XREGS)
    # -----------------------------
    # XREGS1: has strong segment & seasonal structure
    season = np.sin((np.array([d.timetuple().tm_yday for d in dates]) / 365.0) * 2 * np.pi)
    X1 = 200 + 40 * season + 25 * (seg_num - seg_num.mean()) + rng.normal(0, 20, size=n)

    # XREGS2: medium-scale driver, some nonlinearity
    base2 = rng.normal(0, 1, size=n)
    X2 = 100 + 30 * base2 + 10 * (base2 ** 2) + 15 * (seg_num == n_segments) + rng.normal(0, 10, size=n)

    # XREGS3: weak/noisy driver
    X3 = rng.normal(0, 1, size=n) * 50 + rng.normal(0, 30, size=n)

    # -----------------------------
    # 3. Regression target: Leads
    # -----------------------------
    # Non-linear & interaction structure so calibration isn’t trivial
    #   - strong effect from X1 (but saturating)
    #   - modest linear from X2
    #   - small from X3
    #   - segment-specific offsets
    seg_offsets = {
        "S1": -40.0,
        "S2": 0.0,
        "S3": 25.0,
        "S4": 50.0,
        "S5": 75.0,
    }

    seg_offset_arr = np.vectorize(seg_offsets.get)(seg)

    # Saturating transform for X1
    x1_scaled = (X1 - X1.mean()) / X1.std()
    x1_term = 120 * np.tanh(x1_scaled)  # bounded effect

    leads_mean = (
        300
        + x1_term
        + 0.2 * (X2 - X2.mean())
        + 0.05 * X3
        + seg_offset_arr
        + 40 * season  # seasonality in target
    )

    # Heteroskedastic noise: higher noise when X2 is large
    noise_sd = 30 + 0.02 * np.abs(X2)
    Leads = leads_mean + rng.normal(0, noise_sd)

    # Clip to keep numbers nice
    Leads = np.clip(Leads, 0, None)

    # -----------------------------
    # 4. Classification targets from Leads
    # -----------------------------
    # Binary: high lead vs low/medium
    q_hi = np.quantile(Leads, 0.7)
    binary_label = np.where(Leads >= q_hi, 1, 0)

    # Multiclass: quantile buckets of Leads
    q1, q2, q3 = np.quantile(Leads, [0.25, 0.5, 0.75])
    def bucket(y):
        if y < q1:
            return "low"
        elif y < q2:
            return "mid_low"
        elif y < q3:
            return "mid_high"
        else:
            return "high"

    multiclass_label = np.vectorize(bucket)(Leads)

    # For RetroFit examples you can choose which one to use as "Label":
    #  - binary_label for classification
    #  - multiclass_label for multiclass
    # Below we store the multiclass string version; you can overwrite as needed.
    Label = multiclass_label

    # -----------------------------
    # 5. Build Polars DataFrame
    # -----------------------------
    df = pl.DataFrame(
        {
            "CalendarDateColumn": dates,
            "MarketingSegments": seg,
            "MarketingSegments2": seg2,
            "MarketingSegments3": seg3,
            "XREGS1": X1,
            "XREGS2": X2,
            "XREGS3": X3,
            "Leads": Leads,
            "Label": Label,
            "Label_binary": binary_label,   # convenience column
        }
    )

    return df


def do_call(FUN, args=None, kwargs=None):
    """
    Mimic R's do.call() functionality in Python.
    
    Parameters:
        FUN:    The function to be called.
        args:   A list or tuple of positional arguments.
        kwargs: A dictionary of keyword arguments.
    
    Returns:
        The result of calling FUN with the provided arguments.
    
    Example:
        def my_func(alpha, beta):
            return alpha + beta
        
        result = do_call(my_func, args=[0.01], kwargs={'beta': 1})
        print(result)  # Outputs: 1.01
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    return FUN(*args, **kwargs)


def save(x, Path):
    """
    Save a Python object to a file using pickle, similar in spirit to R's save().
    
    Parameters:
        x (any): The object you wish to save.
        Path (str): The file path (including file name). The .pkl extension will be appended
                    if not already present.
    
    Raises:
        ValueError: If Path is not provided.
    
    Example:
        my_object = {"alpha": 0.01, "beta": 1}
        save(my_object, "my_saved_object")
    """
    if Path is None:
        raise ValueError("A valid file path must be provided.")
    
    # Ensure the file name ends with .pkl
    if not Path.endswith('.pkl'):
        Path = f"{Path}.pkl"
    
    try:
        with open(Path, 'wb') as out:
            pickle.dump(x, out, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"An error occurred while saving the object: {e}")


def load(Path):
    """
    Load a Python object from a pickle file, mimicking R's load() functionality.
    
    Parameters:
        Path (str): The file path (including file name). If the path doesn't end with '.pkl',
                    the function will first check for a file with the '.pkl' extension.
    
    Returns:
        The Python object loaded from the pickle file.
    
    Raises:
        ValueError: If no Path is provided.
        FileNotFoundError: If the file is not found.
    """
    if Path is None:
        raise ValueError("A valid file path must be provided.")
    
    # If the provided path doesn't end with '.pkl', try appending it if the file exists.
    if not Path.endswith('.pkl'):
        potential_path = f"{Path}.pkl"
        if os.path.exists(potential_path):
            Path = potential_path
    
    if not os.path.exists(Path):
        raise FileNotFoundError(f"The file '{Path}' does not exist.")
    
    try:
        with open(Path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        raise


def cumsum(x):
    """
    Create a list of summed up values from another list
    """
  
    # Initialize list with prefilled values
    AccumRatios = [z for z in range(len(x))]
  
    # Fill accumulation list
    for acc in range(len(x)):
        if acc == 0:
            AccumRatios[acc] = x[acc]
        else:
            AccumRatios[acc] = AccumRatios[acc-1] + x[acc]

    # Return list
    return(AccumRatios)


def print_dict(x):
    """
    Print out the dictionary where each key : value pair gets a new line
    """
    for z in x:
        if not x[z] is None:
            print(z + ': ' + str(x[z]))
        else:
            print(z + ': None')
