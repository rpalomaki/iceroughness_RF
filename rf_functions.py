import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm.notebook import tqdm



def data_setup(s1_fp, stats_fp=None, moran_fp=None, date=None, s1_units='dB', 
    drop_vv_glcm=True, drop_ad=True):
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
    # Rename VV and VH columns
    s1_data.rename(columns={'vv_raw':'VV', 'vh_raw':'VH'}, inplace=True)
    # Separate targets and predictors
    predict_cols = [col for col in s1_data.columns if re.findall(r'VV|VH', col)]
    target_cols = [col for col in s1_data.columns if col not in predict_cols]
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
    if [c for c in targets.columns if '_10m_' in c]:
        targets['zonal_0219_10m_iqr'] = targets['zonal_0219_10m_p75'] - targets['zonal_0219_10m_p25']
        targets['zonal_0219_10m_p95-p5'] = targets['zonal_0219_10m_p95'] - targets['zonal_0219_10m_p5']
        targets['zonal_0304_10m_iqr'] = targets['zonal_0304_10m_p75'] - targets['zonal_0304_10m_p25']
        targets['zonal_0304_10m_p95-p5'] = targets['zonal_0304_10m_p95'] - targets['zonal_0304_10m_p5']
    

    # Add lognorm fit parameters into targets
    if stats_fp:
        stats_data = pd.read_csv(stats_fp)
        stats_data['date'] = stats_data['date'].astype(str).str.zfill(4)
        stats_data.index = pd.MultiIndex.from_arrays([stats_data['date'], stats_data['S1_pixel_ID']])
        stats_data.drop(columns=['date','S1_pixel_ID'], inplace=True)
        if drop_ad:
            stats_data.drop(columns=['ad'], inplace=True)
        targets = targets.join(stats_data)

    # Add moran data into targets
    if moran_fp:
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


def run_rf_reg(targets, predictors, n_runs=100, rf_type='single_target', 
    train_frac=0.7, rf_params=None, random_state=5033, output_dir_predict=None, 
    output_dir_metrics=None, out_file_prefix=None, return_vals=False):
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
        target_count = 1
        for target_col in targets.columns:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  f'-- Starting target {target_col} ({target_count}/{len(targets.columns)} targets)')
            target_count += 1
            X = predictors
            y = targets[target_col]
            valid_list, predict_list, n_run_list = [], [], []
            rmse_list, mae_list, r2_list = [], [], []
            # If more than one date present, use weighted sampling for t/t split
            # First, create index check object
            ind_check = np.unique(np.array([x[0] for x in targets.index]))
            if len(ind_check) > 1:
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
                if len(ind_check) > 1:
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
                n_run_list.extend(np.repeat(int(i+1), len(predictions)))
                # Metrics
                rmse = metrics.mean_squared_error(y_test, predictions, squared=False) # squared=True -> MSE
                mae = metrics.mean_absolute_error(y_test, predictions)
                r2 = metrics.r2_score(y_test, predictions)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)
            

            predict_df = pd.DataFrame(np.array([n_run_list, predict_list, valid_list]).T, 
                                      columns=['run_no','predict','valid'])
            metrics_df = pd.DataFrame(np.array([rmse_list, mae_list, r2_list]).T, 
                                      columns=['rmse','mae','r2'])
            if output_dir_predict:
                if 'single' in output_dir_predict:
                    date = targets.index[0][0]
                    if date == '0218':
                        date = '0219'
                    fp_p = f'{output_dir_predict}{date}/{out_file_prefix}_{target_col}.csv'
                    fp_m = f'{output_dir_metrics}{date}/{out_file_prefix}_{target_col}.csv'
                else:
                    fp_p = f'{output_dir_predict}{out_file_prefix}_{target_col}.csv'
                    fp_m = f'{output_dir_metrics}{out_file_prefix}_{target_col}.csv'
                predict_df.to_csv(fp_p)
                metrics_df.to_csv(fp_m)

            if return_vals:
                return predict_df, metrics_df
            
    elif rf_type == 'multi_target':
        return None
            



def confusion_heatmap(cm, ax, cmap, vmin=0, vmax=1, norm=True, cbar_label=None,
    xticklabels=None, yticklabels=None,xlabel=True, ylabel=True, font_scale=1,
    fig_title=None, output_dir_cm=None):
    """
    
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
        save_fig = True
    else:
        save_fig = False
    # Normalize if indicated
    if norm:
        cm_df = pd.DataFrame(cm)
        cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0)
        cm = cm_norm.round(2).values
        vmax = 1

    im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(axis='both', which='minor', bottom=False, left=False)
    
    if cbar_label is not None:
        cbar_ratio = cm.shape[0]/cm.shape[1]
        cbar = plt.colorbar(im, ax=ax, fraction=0.046*cbar_ratio, pad=0.04)
        cbar.ax.tick_params(labelsize=10*font_scale)
        if cbar_label: 
            cbar.ax.set_ylabel(cbar_label, rotation=-90, 
                                va='bottom', fontsize=10*font_scale, labelpad=12)

    for i in range(len(cm)):
        for k in range(len(cm)): # cm guaranteed to be square
            if not norm:
                if cm[i, k]/cm.max() > 0.65:
                    textcolor = 'white'
                else:
                    textcolor = 'black'
            else:
                if cm[i, k] > 0.65:
                    textcolor = 'white'
                else:
                    textcolor = 'black'
            
            text = ax.text(k, i, format(cm[i, k], '.2f'), fontsize=15*font_scale,
                            ha='center', va='center', color=textcolor)

    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=11*font_scale)#, rotation=90)
    else:
        ax.set_xticklabels(np.arange(1,len(cm)+1), fontsize=11*font_scale)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=11*font_scale, rotation=90, va='center')
    else:
        ax.set_yticklabels(np.arange(1,len(cm)+1), fontsize=11*font_scale)

    if xlabel: 
        ax.set_xlabel('Predicted label', fontsize=12*font_scale, labelpad=14)
    if ylabel: 
        ax.set_ylabel('True label', fontsize=12*font_scale, labelpad=14)
    if fig_title:
        ax.set_title(fig_title, fontsize=12*font_scale)
    
    plt.tight_layout()
    if save_fig:
        fig.savefig(f"{output_dir_cm}/{fig_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    facecolor='white', dpi=300)
    



def run_rf_cla(targets, predictors, rf_params=None, n_runs=100, train_frac=0.7, 
    classes=5, class_split_method='percentile', plot_cm =True, 
    random_state=5033, output_dir_predict=None, output_dir_cm=None, 
    out_file_prefix=None, return_cm=False):
    """
    stuff
    """
# Some initial setup
    target_count = 1
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          f'-- Starting RF classification ({len(targets.columns)} targets total).')
    for target_col in targets.columns:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f'-- Starting target {target_col} ({target_count}/{len(targets.columns)} targets)')
        target_count += 1
        # Create classes
        X = predictors
    
        if class_split_method == 'percentile':
            target_raw = targets[target_col]
            if type(classes) == int:
                # Calculate equal-width percentile bounds based on n_classes
                p_bounds = np.linspace(0, 100, classes+1)
                percentiles = [np.percentile(target_raw, p) for p in p_bounds]
                class_labels = np.arange(1, len(p_bounds))
            elif type(classes) == list and class_split_method == 'percentile':
                # Use specified list of percentiles
                p_bounds = classes
                percentiles = [np.percentile(target_raw, p) for p in p_bounds]
                class_labels = np.arange(1, len(p_bounds))
            
            y = pd.cut(target_raw, bins=percentiles, labels=class_labels)
            # NaN check
            y.dropna(inplace=True)
            X = X.loc[y.index]

        elif class_split_method == 'custom':
            y = targets[target_col]

        else:
            # Reserve for other split methods methods
            raise ValueError("class_split_method must be either 'percentile' or 'custom'.")
        
        valid_list, predict_list, n_run_list = [], [], []
        # If more than one date present, use weighted sampling for t/t split
        # First, create index check object
        ind_check = np.unique(np.array([x[0] for x in targets.index]))
        if len(ind_check) > 1:
            nsamples_0218 = int(targets.loc['0218'].shape[0]*train_frac)
            nsamples_0302 = int(targets.loc['0302'].shape[0]*train_frac)
        
        # Initiate RandomForestClassifier
        if rf_params is None:
            # Default parameters
            ntrees = 2000    # 2000 recommended in 10.5194/hess-25-2997-2021
            mtry = 2/3       # From paper above; passed to 'max_features' kwarg in RF regressor
            max_depth = 7    # Max depth of each tree
            rf = RandomForestClassifier(n_estimators=ntrees,
                            max_features=mtry,
                            max_depth=max_depth, 
                            random_state=random_state)
        else:
            try:
                rf = RandomForestClassifier(**rf_params)
            except TypeError:
                print('rf_params must be a dict with valid arguments \
                        for the sklearn RandomForestClassifier class.')
                return None
            
        
        # Loop of RF runs
        for i in tqdm(range(n_runs)):
            # Train/test split
            if len(ind_check) > 1:
                # Weighted sampling of multiple dates
                train_ind_0218 = pd.MultiIndex.from_product([['0218'], X.loc['0218'].sample(nsamples_0218).index])
                train_ind_0302 = pd.MultiIndex.from_product([['0302'], X.loc['0302'].sample(nsamples_0302).index])
                train_ind = train_ind_0218.union(train_ind_0302)
                test_ind = X[~X.index.isin(train_ind)].index
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
            n_run_list.extend(np.repeat(int(i+1), len(predictions)))

        predict_df = pd.DataFrame(np.array([n_run_list, predict_list, valid_list]).T, 
                                    columns=['run_no','predict','valid'])
        
        if output_dir_predict:
            if 'single' in output_dir_predict:
                date = targets.index[0][0]
                if date == '0218':
                    date = '0219'
                fp_p = f'{output_dir_predict}{date}/{out_file_prefix}_{target_col}.csv'
            else:
                fp_p = f'{output_dir_predict}{out_file_prefix}_{target_col}.csv'
            predict_df.to_csv(fp_p)

        if plot_cm:
            # Confusion matrix
            cm_labels = np.unique(predict_list).tolist()
            cm = metrics.confusion_matrix(y_true=valid_list, y_pred=predict_list, 
                                          labels=cm_labels)
            fig, ax = plt.subplots(figsize=(6,6))
            if 'single' in output_dir_predict:
                fig_title = f'{target_col}_single_{date}'
            else:
                fig_title = f'{target_col}_multi'
            confusion_heatmap(cm, ax=None, cmap='Blues', vmin=0, vmax=cm.max(),
                              norm=True,
                              xticklabels=cm_labels, yticklabels=cm_labels,
                              cbar_label='Percentage', fig_title=fig_title, 
                              output_dir_cm=output_dir_cm)
            plt.close('all')

            if return_cm:
                return cm, cm_labels