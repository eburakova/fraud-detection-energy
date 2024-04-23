### run this file from terminal with python3 create_df.py 
### find the merged_df in your data folder

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
print('+++ This takes some time, do not give up +++')

# ge the data 
df_client = pd.read_csv('data/client_train.csv', parse_dates=['creation_date'], dayfirst=True, low_memory=False)
df_invoice = pd.read_csv('data/invoice_train.csv', parse_dates=['invoice_date'], dayfirst=True, low_memory=False)

# rename columns/ids
df_client.rename({'disrict': 'district'}, axis=1, inplace=True)
df_invoice['client_id'] = df_invoice.client_id.str.removeprefix('train_Client_').astype(int)
df_client['client_id'] = df_client.client_id.str.removeprefix('train_Client_').astype(int)

# drop duplicates
df_invoice.drop_duplicates(inplace=True)

##########################
###### df_client #########
##########################

# convert data types 
def convert_feature_types_to_category(client_data, cat_cols, dtype='object'):
    # convert categorical features in client_df to categories 
    for col in cat_cols:
        client_data [col] = client_data[col].astype(dtype)
    return client_data

client_data_cat_cols = ['region','client_catg', 'district', 'target']
df_client = convert_feature_types_to_category(df_client, client_data_cat_cols)

# create datetime features from client df
df_client['acc_creation_year'] = df_client.creation_date.dt.year
df_client['acc_creation_month'] = df_client.creation_date.dt.month
df_client['acc_creation_weekday'] = df_client.creation_date.dt.dayofweek

##########################
###### df_invoice ######## - feature engineering
##########################

# create datetime features from client invoice data by years, months and weekdays  
df_invoice['invoice_year'] = df_invoice.invoice_date.dt.year
df_invoice['invoice_month'] = df_invoice.invoice_date.dt.month
df_invoice['invoice_weekday'] = df_invoice.invoice_date.dt.dayofweek
 
######################
# consumption levels # - of gas and energy
######################

# calculate mean and std, min, max over both counter types (total) and elec and gaz seperately
# for each consumption level for each client

# sub df by counter_type for further calculations
df_elec = df_invoice[df_invoice.counter_type == 'ELEC']
df_gaz = df_invoice[df_invoice.counter_type == 'GAZ']

def aggregate_consumption_data(merged_df, invoice_data, col_prefix): 
    agg_operations = ['mean', 'std', 'max', 'min']
    aggs = {}
    for operation in agg_operations:
        aggs['consommation_level_1'] = [operation]
        aggs['consommation_level_2'] = [operation]
        aggs['consommation_level_3'] = [operation]
        aggs['consommation_level_4'] = [operation]
        # aggregating
        df_consumption = invoice_data.groupby(['client_id'], as_index=False).agg(aggs)
        # renaming
        df_consumption.columns = ['_'.join(col).strip() for col in df_consumption.columns.values]
        df_consumption = df_consumption.add_prefix(col_prefix) 
        df_consumption.columns = [col.replace('consommation_level_','consum_lvl') for col in df_consumption.columns]
        df_consumption.rename({f'{col_prefix}client_id_': f'client_id'}, axis=1, inplace=True) 
        # merging
        merged_df  = pd.merge(merged_df, df_consumption, on='client_id', how='outer')
    return merged_df

# aggregate over both counter types
df_merged = aggregate_consumption_data(df_client, df_invoice, 'total_')
# aggregate only for elec
df_merged = aggregate_consumption_data(df_merged, df_elec, 'elec_')
# aggregate only for gas
df_merged = aggregate_consumption_data(df_merged, df_gaz, 'gaz_')


###########################################
### mode and count categorical features ###
###########################################

# functions to create a mode and count feature of a categorical variable 
# for each client_id and merge them in a df


def create_mode_feature(invoice_data: pd.DataFrame, feature: str, renamed_feature: str):
    # create a new feature with feature mode for each client
    feature_mode = invoice_data.groupby('client_id', as_index=False)[feature].agg({'column': lambda x: pd.Series.mode(x)[0]}) # mode return always a series, only return first value
    feature_mode.rename(columns={'column': f'{renamed_feature}_mode'}, inplace=True)
    return feature_mode


def create_count_feature(invoice_data: pd.DataFrame, feature: str, verbose=0):
    # check if there are more than one feature category per client
    # print info with verbose = 1
    feature_count = invoice_data.groupby('client_id', as_index=False)[feature].nunique().sort_values(feature)
    if verbose:
        print(f'There are {len(feature_count[feature_count[feature] == 1])} clients with only 1 {feature}.')
        print(f'There are {len(feature_count[feature_count[feature] > 1])} clients with more than 1 {feature}.')
        print(f'The max number of different {feature}s per client is {feature_count[feature].max()}.')
    return feature_count


def merge_features(feature_mode: pd.DataFrame, feature_count : pd.DataFrame, feature: str, renamed_feature: str):
    # merge feature_count to feature_mode
    df_features = pd.merge(feature_mode,feature_count, on='client_id', how='outer')
    df_features.rename(columns={feature : f'{renamed_feature}_count'}, inplace=True)
    return df_features


def create_feature_df(invoice_data: pd.DataFrame, feature: str, renamed_feature: str):
    # function takes the 3 functions above and returns a df with the merged features
    feature_mode = create_mode_feature(invoice_data, feature, renamed_feature)
    feature_count = create_count_feature(invoice_data, feature)
    return merge_features(feature_mode, feature_count, feature ,renamed_feature)

####################
### new features ###
####################

print('+++ Calculating the modes +++')
# energy tarif type
df_elec['tarif_type'] = df_elec['tarif_type'].astype(str).copy()
elec_tarif_type_features = create_feature_df(df_elec,'tarif_type', 'energy_tarif_type')

# gas tarif type
df_gaz['tarif_type'] = df_gaz['tarif_type'].astype(str).copy()
gaz_tarif_type_features = create_feature_df(df_gaz,'tarif_type', 'gas_tarif_type')

# counter status 
df_invoice['counter_statue'] = df_invoice['counter_statue'].astype(str)
# recode all 'odd' counter_statue codes with a 99 as category for 'other'
scodes_to_recode = ['46', 'A', '618', '769', '269375', '420']
df_invoice['counter_statue'].replace(scodes_to_recode, '99', inplace=True)
counter_statue_features = create_feature_df(df_invoice,'counter_statue', 'counter_status')

# counter_code
df_invoice['counter_code'] = df_invoice['counter_code'].astype(str)
counter_code_features = create_feature_df(df_invoice,'counter_code', 'counter_code')

# reading_remarque
df_invoice['reading_remarque'] = df_invoice['reading_remarque'].astype(str)
# recode all 'odd' reading remarks with 99 as category for 'other'
codes_to_recode = ['5', '207', '413', '203']
df_invoice['reading_remarque'].replace(codes_to_recode, '99', inplace=True)
remark_features = create_feature_df(df_invoice,'reading_remarque', 'remark')

# counter_number
counter_nr_features = create_feature_df(df_invoice,'counter_number', 'counter_number')

# counter_coefficient
# because most of the coefficients occur only once, try putting them together in an'other' category
ccodes_to_recode = ['5', '11', '8', '33', '50', '9', '20', '10', '4']
sub_df = df_invoice[df_invoice.counter_coefficient.isin(ccodes_to_recode)]
# recode all 'odd' reading remarks with 99 as category for 'other'
df_invoice['counter_coefficient'].replace(ccodes_to_recode, '99', inplace=True)
counter_coeff_features = create_feature_df(df_invoice,'counter_coefficient', 'counter_coeff')

#####################################
### invoice date related features ###
#####################################

# account duration (in years for gas and energy and the abs diff between both :)
def extract_account_duration(df_by_counter_type, prefix=''):
    df_time_diff = df_by_counter_type.sort_values('invoice_date').groupby('client_id', as_index=False)['invoice_date'].agg(['first','last'])
    df_time_diff[f'{prefix}acc_dur_years'] = (df_time_diff['last'] - df_time_diff['first']) / np.timedelta64(1, 'Y')
    df_time_diff.drop(['first', 'last'], axis=1, inplace=True)
    return df_time_diff

elect_acc_dur = extract_account_duration(df_elec, prefix='elec_')
gaz_acc_dur = extract_account_duration(df_gaz, prefix='gaz_')
features_account_dur = pd.merge(elect_acc_dur,gaz_acc_dur, on='client_id', how='outer')
features_account_dur['difference_acc_dur']= abs(features_account_dur.elec_acc_dur_years - features_account_dur.gaz_acc_dur_years)

# montly consum for all levels and both counter types
def calculate_montly_consum(counter_type_data, type, level):
    montly_consum = counter_type_data.groupby(['client_id', 'invoice_month'], as_index=False)[f'consommation_level_{level}'].mean()
    montly_consum_wide = montly_consum.pivot(index='client_id', columns='invoice_month', values=f'consommation_level_{level}')
    montly_consum_wide.columns = [f'{type}_{level}_mon_{column}' for column in montly_consum_wide.columns]
    montly_consum_wide = montly_consum_wide.reset_index()
    return montly_consum_wide

#################################
### add features to df_merged ###
#################################

print('+++ Merging dfs +++')

features_types = [gaz_tarif_type_features, elec_tarif_type_features, counter_statue_features, counter_code_features, remark_features, counter_nr_features, counter_coeff_features, features_account_dur] 
for feature in features_types:
    df_merged = pd.merge(df_merged, feature, on='client_id', how='outer')    

# for monthly consumption of energy
for levels in range(1,5):
    montly_consum=calculate_montly_consum(df_elec, 'elec', levels) 
    df_merged = pd.merge(df_merged, montly_consum, on='client_id', how='outer')
    
# for monthly consumption of gas
# two last consumption levels for gas are empty.
# The second level has too few data points (up to 20 readings) 
# Hence it's better to aggregate for the first level only.
for levels in range(1,2):
    montly_consum=calculate_montly_consum(df_gaz, 'gaz', levels) 
    df_merged = pd.merge(df_merged, montly_consum, on='client_id', how='outer') 

############################
### change feature types ###
############################
    
# set categories to merged mode features (keep missing values to recode as a separate category)
categorical_features = ['energy_tarif_type_mode','gas_tarif_type_mode', 'counter_status_mode', 'counter_code_mode','remark_mode', 'counter_number_mode','counter_coeff_mode']
convert_feature_types_to_category(df_merged, categorical_features, dtype='object')
df_merged[categorical_features].fillna('missing',inplace=True)

# set int to merged count features (and replace missing values with 0)
int_features = ['energy_tarif_type_count','gas_tarif_type_count', 'counter_status_count', 'counter_code_count','remark_count', 'counter_number_count', 'counter_coeff_count']
#for feature in  int_features:
df_merged[int_features].fillna(0,inplace=True)
#convert_feature_types_to_category(df_merged, int_features, dtype='int')

############################
###  drop empty columns  ###
############################

# Gas has only two consumption levels, drop unnecessary features
# gas_cols = ['gaz_3_mon_'+str(mon) for mon in range(1,13)] + ['gaz_4_mon_'+str(mon) for mon in range(1,13)]
# df_merged.drop(gas_cols, axis=1, inplace=True)

#######################
###### merged_df ######
#######################

# replace NaN with 0 
#df_merged.fillna({col: 0.0 for col in df_merged.columns[df_merged.dtypes.eq(float)]}, inplace=True)

# attention **important note**: there are many missing values due to aggregating and merging!
# I cannot simply set nas to 0 in categorical variables (that would change the category or add a completely new one)
# that maskes no sense here therefore I'll leave them there and you have to take care of them when imputing as a preprocessing step !!

# save df as csv
print('+++ Success! Saving merged_df.csv file in data +++')
df_merged.to_csv('data/merged_df_full.csv', index=False) 

