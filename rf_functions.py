import pandas as pd



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