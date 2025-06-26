import pandas as pd
import numpy as np
import scipy.stats


def fit_distribution(actuals, dist_name=None, min_val=None, max_val=None, verbose=False):

    # Filter actuals based on min and max values if provided to match simulation boundries
    if min_val is not None:
        actuals = actuals[actuals >= min_val]
    if max_val is not None:
        actuals = actuals[actuals <= max_val]

    all_dists_df = pd.DataFrame(
        columns=['Distribution', 'MLE_Params', 'K-S_Stat', 'K-S_p-value', 'AD_Stat', 'Param1', 'Param2', 'Param3',
                 'Param_Names'])
    temp_dfs = []

    param_names_dict = {
        'norm': ['Mean', 'Std Dev'],
        'lognorm': ['Shape (s)', 'Loc', 'Scale'],
        'gamma': ['Shape (a)', 'Loc', 'Scale'],
        'weibull_min': ['Shape (c)', 'Loc', 'Scale']
    }

    if dist_name is None:
        dists = ['norm', 'lognorm', 'gamma', 'weibull_min']

        for dist_name in dists:
            dist = getattr(scipy.stats, dist_name)
            params = dist.fit(actuals)

            # Perform K-S test
            D, p_value = scipy.stats.kstest(actuals, dist_name, args=params)

            # Extract individual parameters for separate columns
            param1, param2, param3 = params if len(params) == 3 else (*params, None)

            # Get parameter names
            param_names = param_names_dict.get(dist_name, ['-'])
            param_names_str = ', '.join([f"{i + 1}:{name}" for i, name in enumerate(param_names)])

            # Add to temporary DataFrame list
            temp_df = pd.DataFrame({
                'Distribution': [dist_name],
                # 'MLE_Params': [params],
                'Param_Names': [param_names_str],
                'Param1': [param1],
                'Param2': [param2],
                'Param3': [param3],
                'K-S_Stat': [D],
                'K-S_p-value': [p_value],
            })
            temp_dfs.append(temp_df)

        # Concatenate all temporary DataFrames
        all_dists_df = pd.concat(temp_dfs, ignore_index=True)

        # Find the best distribution based on K-S test p-value
        dist_name = all_dists_df.loc[all_dists_df['K-S_p-value'].idxmax(), 'Distribution']

    # Fit the best distribution to the data
    dist = getattr(scipy.stats, dist_name)
    params = dist.fit(actuals)

    if verbose:
        print(f"Fitting distribution: {dist_name}")
        print(f"MLE Parameters: {params}")

    return dist_name, params, all_dists_df


def simulate_actuals(actuals, dist_name, params, samples=2800, min_val=None, max_val=None, verbose=False):

    # set min max
    max_actual = np.max(actuals) if max_val is None else max_val
    min_actual = np.min(actuals) if min_val is None else min_val

    # setup dist and scipy
    dist = getattr(scipy.stats, dist_name)

    if min_val is not None or max_val is not None:
        # Manual truncation: rejection sampling
        simulated_samples = []
        while len(simulated_samples) < samples:
            sample = dist.rvs(*params, size=samples)
            # Only keep samples within the min and max bounds
            valid_samples = sample[(sample >= min_actual) & (sample <= max_actual)]
            simulated_samples.extend(valid_samples)
        simulated_samples = np.array(simulated_samples[:samples])
    else:
        simulated_samples = dist.rvs(size=samples, *params)

    if verbose:
        print(f"Best-fitting distribution: {dist_name}")
        print(f"MLE Parameters: {params}")
        if min_val is not None or max_val is not None:
            print(f"Truncating distribution between {min_actual} and {max_actual}")

    return simulated_samples