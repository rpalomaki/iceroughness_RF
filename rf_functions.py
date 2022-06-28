import pandas as pd



def data_setup(s1_fp, s1_units='db', drop_vv_glcm=True):
    """

    Arguments
    ---------
    s1_fp : str
        stuff
    s1_units : 'db' or 'raw_power'
        stuff
    drop_vv_glcm : bool (default True)
        stuff
    
    
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
