################################################
# IMPORT
################################################
import pandas as pd
import numpy as np

################################################
# LOAD DATA
################################################
data_processed_location = 'data/processed/'
processed_data_filename = 'df_application_record_classified_raw.pickle'
feat_eng_processed_data_filename = 'df_application_record_classified_post_feature_eng.pickle'

df_merged = pd.read_pickle(data_processed_location + processed_data_filename)

################################################
# FEATURE ENGINEERING
################################################
df_merged['amt_income_per_person'] = df_merged['amt_income_total'] / df_merged['cnt_fam_members']
df_merged['age'] = -1 * df_merged['days_birth'] / 365
df_merged['job_tenure'] = -1 * df_merged['days_employed'] / 365
df_merged['job_tenure'] = np.where(df_merged["days_employed"] > 0, -1, df_merged['job_tenure'])
df_merged['flag_employed'] = np.where(df_merged["days_employed"] > 0, 0, 1)
df_merged['code_gender'] = np.where(df_merged['code_gender'] == 'M', 1, 0)
df_merged['flag_own_car'] = np.where(df_merged['flag_own_car'] == 'Y', 1, 0)
df_merged['flag_own_realty'] = np.where(df_merged['flag_own_realty'] == 'Y', 1, 0)
df_merged['single_adult'] = np.where(
    (df_merged['name_family_status'] == 'Married') | (df_merged['name_family_status'] == 'Civil marriage'), 0, 1)
df_merged['target'] = np.where((df_merged['customer_status'] == 'bad customer') | (df_merged['customer_status'] == 'moderate customer'), 1, 0)
df_merged = df_merged.astype({'flag_own_car': int,
                              'flag_own_realty': int,
                              'cnt_children': int,
                              'cnt_fam_members': int})

################################################
# FEATURES SELECTION
################################################
df_merged = df_merged.drop(['flag_mobil', 'flag_work_phone', 'flag_phone',
                            'flag_email', 'days_birth', 'days_employed',
                            'name_family_status', 'customer_status'], axis=1)

################################################
# SAVE DATA
################################################
df_merged.to_pickle(data_processed_location + feat_eng_processed_data_filename)
