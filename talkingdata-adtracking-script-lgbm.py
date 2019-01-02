# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import numpy as np # linear algebra
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.environ['OMP_NUM_THREADS'] = '4'

debug=0  #Whethere or not in debuging mode

nrows=184903891-1
nchunk=25000000   
val_size=2500000
predictors=[]

frm=nrows-65000000
if debug:
    frm=0
    nchunk=100000
    val_size=10000

to=frm+nchunk

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 10,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.9,  # L1 regularization term on weights
        'reg_lambda': 0.9,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("bst.best_iteration: ", bst1.best_iteration)
    print('auc'+":", evals_results['valid']['auc'][bst1.best_iteration-1])

    return bst1


def do_countuniq( df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return( df )
    
def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)
    
# Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return( df )
    
path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        

        
print('loading train data...')
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, parse_dates=['click_time'], usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
if debug:
    test_df = pd.read_csv(path+"test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
else:
    test_df = pd.read_csv(path+"test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')
train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('int8')

gc.collect()


print('cal freq : ip ,app, device, os, channel')
FREQUENCY_COLUMNS = ['ip', 'app', 'channel'] # 'ip', 'app', 'device', 'os', 'channel'

# Find frequency of is_attributed for each unique value in column
freqs = {}
for col in FREQUENCY_COLUMNS:
    print(f">> Calculating frequency for: {col}")

    # Get counts, sums and frequency of is_attributed
    df = pd.DataFrame({
        'sums': train_df.groupby(col)['is_attributed'].sum(),
        'counts': train_df.groupby(col)['is_attributed'].count()
    })
    df.loc[:, 'freq'] = df.sums / df.counts
    
    # If we have less than 3 observations, e.g. for an IP, then assume freq of 0
    df.loc[df.counts <= 3, 'freq'] = 0        
    
    # Add to X_total
    train_df[col+'_freq'] = train_df[col].map(df['freq'])
    predictors.append(col+'_freq')
    del df
    gc.collect()

print('grouping by ip-day-hour combination...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
predictors.append('ip_tcount')
del gp
gc.collect()
train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')

print('grouping by ip-app combination...')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
predictors.append('ip_app_count')
del gp
gc.collect()
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')


print('grouping by ip-app-os combination...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
predictors.append('ip_app_os_count')
del gp
gc.collect()
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')


# # Adding features with var and mean hour (inspired from nuhsikander's script)
# print('grouping by : ip_day_chl_var_hour')
# gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
# train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
# predictors.append('ip_tchan_count')
# del gp
# gc.collect()

# print('grouping by : ip_app_os_var_hour')
# gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
# train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
# predictors.append('ip_app_os_var')
# del gp
# gc.collect()

# print('grouping by : ip_app_channel_var_day')
# gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
# train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
# predictors.append('ip_app_channel_var_day')
# del gp
# gc.collect()

# print('grouping by : ip_app_chl_mean_hour')
# gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
# print("merging...")
# train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
# predictors.append('ip_app_channel_mean_hour')
# del gp
# gc.collect()

print("vars and data type: ")
train_df.info()
    
train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'channel' ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'day'], 'hour' ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os'); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device'); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel'); gc.collect()
train_df = do_count( train_df, ['ip', 'day', 'hour'] ); gc.collect()
train_df = do_count( train_df, ['ip', 'app']); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os']); gc.collect()


test_df = train_df[len_train:]
val_df = train_df[(len_train-val_size):len_train]
train_df = train_df[:(len_train-val_size)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'


categorical = ['app', 'device', 'os', 'channel', 'hour', 'minute']  # , 'day', 'wday'

for feature in categorical:
    if feature not in predictors:
        predictors.append(feature)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
start_time = time.time()


params = {
    'learning_rate': 0.01,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 100,  # 2^max_depth - 1
    'max_depth': 7,  # -1 means no limit
    'min_child_samples': 200,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 250,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 99.74, # because training data is extremely unbalanced 
    'reg_alpha': 0.9,  # L1 regularization term on weights
    'reg_lambda': 0.9,  # L2 regularization term on weights
    'early_stopping_round': 5,
}



# # Create parameters to search
# gridParams = {
#     'learning_rate': [0.005],
#     'n_estimators': [8,16,24],
#     'num_leaves': [6,8,12,16],
#     'boosting_type' : ['gbdt'],
#     'objective' : ['binary'],
#     'random_state' : [501], # Updated from 'seed'
#     # 'colsample_bytree' : [0.64, 0.65, 0.66],
#     'subsample' : [0.7,0.75],
#     'reg_alpha' : [1,1.2],
#     'reg_lambda' : [1,1.2,1.4],
#     }
# 
# # Create classifier to use. Note that parameters have to be input manually
# # not as a dict!
# mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
#           objective = 'binary', 
#           n_jobs = 5, # Updated from 'nthread' 
#           silent = True,
#           max_depth = params['max_depth'],
#           max_bin = params['max_bin'], 
#         #   subsample_for_bin = params['subsample_for_bin'],
#           subsample = params['subsample'], 
#           subsample_freq = params['subsample_freq'], 
#         #   min_split_gain = params['min_split_gain'], 
#           min_child_weight = params['min_child_weight'], 
#           min_child_samples = params['min_child_samples'], 
#           scale_pos_weight = params['scale_pos_weight'])

# # To view the default model params:
# mdl.get_params().keys()

# # Create the grid
# grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
# # Run the grid
# train_for_grid = train_df.drop(['click_id', 'click_time','ip','is_attributed'],1)
# grid.fit(train_for_grid, train_df[target].values)

# # Print the best parameters found
# print(grid.best_params_)
# print(grid.best_score_)

# # Using parameters already set above, replace in the best from the grid search
# # params['colsample_bytree'] = grid.best_params_['colsample_bytree']
# params['learning_rate'] = grid.best_params_['learning_rate'] 
# # params['max_bin'] = grid.best_params_['max_bin']
# params['num_leaves'] = grid.best_params_['num_leaves']
# params['reg_alpha'] = grid.best_params_['reg_alpha']
# params['reg_lambda'] = grid.best_params_['reg_lambda']
# params['subsample'] = grid.best_params_['subsample']
# # params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

# print('Fitting with params: ')
# print(params)


bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=1000, 
                        categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))
del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")


print("Feature importance")
for name in bst.feature_name():
    print("Importance for {} is {}".format(name, bst.feature_importance()[bst.feature_name().index(name)]))
    
ax = lgb.plot_importance(bst, max_num_features=100)
plt.show()
plt.savefig('test.png', dpi=600,bbox_inches="tight")   