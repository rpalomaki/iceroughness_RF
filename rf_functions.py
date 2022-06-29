import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm.notebook import tqdm



def data_setup(s1_fp, stats_fp, moran_fp, date=None, s1_units='dB', drop_vv_glcm=True,
               drop_ad=True):
    """
    Function to compile all input data to run icer roughness RF models.

    Arguments
    ---------
    s1_fp : str
        Filepath to Sentinel 1 data.
    stats_fp : str
        Filepath to statistics data.
    moran_fp : str
        Filepath to Moran's I data.
    date : str, default None
        A date used to subset the larger dataset. If None, returns all 
        available data.
    s1_units : 'dB' (default) or 'raw_power'
        If 'dB', converts raw VV and VH Sentinel 1 returns to dB scale.
    drop_vv_glcm : bool, default True
        Controls whether or not to drop VV GLCM data (currently, VV GLCM does 
        not produce meaningful results).
    drop_ad : bool, default True
        Controls whether or not to drop Anderson-Darling test data from the 
        statistics dataset. Currently AD test does not produce meaningful 
        results.

    Returns
    -------
    targets : pd.Dataframe
        DataFrame containing the possible RF targets.
    predictors : pd.DataFrame
        DataFrame containing the possible RF predictors.
    """
    # Read/tidy S1 data
    s1_data = pd.read_csv(s1_fp)
    s1_data['date'] = s1_data['date'].astype(str).str.zfill(4)
    s1_data.index = pd.MultiIndex.from_arrays([s1_data['date'], s1_data['S1_pixel_ID']])
    s1_data.drop(columns=['date','S1_pixel_ID'], inplace=True)
    s1_data.sort_index(inplace=True)
    # Separate targets and predictors
    target_cols = [col for col in s1_data.columns if 'roughness' in col]
    predict_cols = [col for col in s1_data.columns if col not in target_cols]
    targets = s1_data[target_cols]
    predictors = s1_data[predict_cols]

    # Create derived metrics - predictors
    if s1_units == 'dB':
        predictors['VV'] = 10*np.log10(predictors['VV'])
        predictors['VH'] = 10*np.log10(predictors['VH'])
    if drop_vv_glcm:
        predictors.drop(columns=[col for col in predictors.columns if 'VV_GLCM' in col], inplace=True)
    predictors['vv_vh_ratio'] = predictors['VV']/predictors['VH']
    predictors['vh_vv_ratio'] = predictors['VH']/predictors['VV']
    predictors['vv2'] = predictors['VV']**2
    predictors['vh2'] = predictors['VH']**2
    predictors['vv_inv'] = 1/predictors['VV']
    predictors['vh_inv'] = 1/predictors['VH']
    predictors['multiply'] = predictors['VV'] * predictors['VH']

    # Create derived metrics - targets
    targets['iqr'] = targets['roughness_p75'] - targets['roughness_p25']
    targets['p95-p5'] = targets['roughness_p95'] - targets['roughness_p5']

    # Add lognorm fit parameters into targets
    stats_data = pd.read_csv(stats_fp)
    stats_data['date'] = stats_data['date'].astype(str).str.zfill(4)
    stats_data.index = pd.MultiIndex.from_arrays([stats_data['date'], stats_data['S1_pixel_ID']])
    stats_data.drop(columns=['date','S1_pixel_ID'], inplace=True)
    if drop_ad:
        stats_data.drop(columns=['ad'], inplace=True)
    targets = targets.join(stats_data)

    # Add moran data into targets
    moran = pd.read_csv(moran_fp)
    moran['date'] = moran['date'].astype(str).str.zfill(4)
    moran.index = pd.MultiIndex.from_arrays([moran['date'], moran['S1_pixel_ID']])
    moran.drop(columns=['date','S1_pixel_ID'], inplace=True)
    moran.columns = ['moran','moran_z','moran_p']
    targets = targets.join(moran)

    # Subset by date if necessary
    if date == '0218':
        targets = targets.loc['0218']
        predictors = predictors.loc['0218']
    elif date == '0302':
        targets = targets.loc['0302']
        predictors = predictors.loc['0302']
    
    return targets, predictors


def run_rf(targets, predictors, n_runs=100, rf_type='single_target', 
           train_frac=0.7, rf_params=None, random_state=5033, 
           output_dir=None, out_file_prefix=None):
    """
    
    Arguments
    ---------
    targets : pd.DataFrame
        The target(s) to use for the RF model.
    predictors : pd.DataFrame
        The predictors to use for the RF model.
    rf_type : 'single_target' (default) or 'multi_target'
        If 'single_target', this function will iterate through all columns in 
        `targets` df and run RF regression on each column as a separate target.
        If 'multi_target', this function will consider all columns in `targets`
        simultaneously as one n-dimensional target.
    train_frac : float
        stuff
    rf_params : dict, default None
        If None, calls RandomForestRegressor passing parameters as follows:
        RandomForestRegressor(n_estimators=ntrees,
                              max_features=mtry,
                              max_depth=max_depth, 
                              random_state=random_state)
        If dict, unpacks k, v pairs to pass as params to the RFR call.
    random_state : int
        Random state for initializing RandomForestRegressor. Does not get 
        passed to train/test split function. This random state will be 
        overwritten if a `random_state` kwarg is passed as part of the 
        `rf_params` dictionary.
    output_fp : str, default None.
        If not none, the filepath to save the RF data using DataFrame.to_csv.
    
    """
    # RF analysis - single target column
    if rf_type == 'single_target':
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              f'-- Starting single target RF regression ({len(targets.columns)} targets total).')
        
        # Some initial setup
        for target_col in targets.columns:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  f'-- Starting target {target_col}')
            X = predictors
            y = targets[target_col]
            valid_list, predict_list, n_run_list = [], [], []
            # If more than one date present, use weighted sampling for t/t split
            if len(targets.index.levels[0]) > 1:
                nsamples_0218 = int(targets.loc['0218'].shape[0]*train_frac)
                nsamples_0302 = int(targets.loc['0302'].shape[0]*train_frac)
            
            # Initiate RandomForestRegressor
            if rf_params is None:
                # Default parameters
                ntrees = 2000    # 2000 recommended in 10.5194/hess-25-2997-2021
                mtry = 2/3       # From paper above; passed to 'max_features' kwarg in RF regressor
                max_depth = 7    # Max depth of each tree
                rf = RandomForestRegressor(n_estimators=ntrees,
                                max_features=mtry,
                                max_depth=max_depth, 
                                random_state=random_state)
            else:
                try:
                    rf = RandomForestRegressor(**rf_params)
                except TypeError:
                    print('rf_params must be a dict with valid arguments \
                           for the sklearn RandomForestRegressor class.')
                    return None
                
            
            # Loop of RF runs
            for i in tqdm(range(n_runs)):
                # Train/test split
                if len(targets.index.levels[0]) > 1:
                    # Weighted sampling of multiple dates
                    train_ind_0218 = pd.MultiIndex.from_product([['0218'], targets.loc['0218'].sample(nsamples_0218).index])
                    train_ind_0302 = pd.MultiIndex.from_product([['0302'], targets.loc['0302'].sample(nsamples_0302).index])
                    train_ind = train_ind_0218.union(train_ind_0302)
                    test_ind = targets[~targets.index.isin(train_ind)].index
                    X_train, X_test = X.loc[train_ind], X.loc[test_ind]
                    y_train, y_test = y.loc[train_ind], y.loc[test_ind]
                else:
                    # Single date only
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_frac)
                # Fit and predict
                rf.fit(X_train, y_train)
                predictions = rf.predict(X_test)
                # Save values to list
                valid_list.extend(y_test.values.tolist())
                predict_list.extend(predictions)
                n_run_list.extend(np.repeat(i+1, len(predictions)))
                # if i != 0 and not (i+1)%100:
                #     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                #         f'-- {i+1}/{n_runs} complete.')
            

            output = pd.DataFrame(np.array([n_run_list, predict_list, valid_list]).T, 
                                    columns=['run_no','predict','valid'])
            if output_dir:
                output.to_csv(output_dir + out_file_prefix + f'_{target_col}.csv')
            
            return output
            
    elif rf_type == 'multi_target':
        return None
            



# def plot_results():
#     """
    
#     """
#     plt.close('all')
#     fig, ax = plt.subplots()
#     hex1 = ax.hexbin(predict_list, valid_list, gridsize=50, cmap='inferno')
#     ax.plot((min(predict_list), max(predict_list)), (min(predict_list), max(predict_list)), 'r--')
#     cb1 = fig.colorbar(hex1, ax=ax)
#     cb1.set_label(f'Counts ({len(predict_list)} total predictions)', labelpad=12)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('Measured', labelpad=12)
#     ax.set_title(f'{target_col} of 10m pixels trained on both dates')
#     plt.tight_layout()
#     if output_figures:
#         fig.savefig(hex_figure_fname, dpi=300)
