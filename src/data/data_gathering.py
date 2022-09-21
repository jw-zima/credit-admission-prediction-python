################################################
# IMPORT
################################################
import pandas as pd
import numpy as np

################################################
# LOAD DATA
################################################
def lower_cols(df):
    df.columns = df.columns.str.lower()
    return df


df_application_record = pd.read_csv("data/raw/application_record.csv")
df_application_record = lower_cols(df_application_record)


df_credit_record = pd.read_csv("data/raw/credit_record.csv")
df_credit_record = lower_cols(df_credit_record)

################################################
# CREDIT RECORD DATA - AGGREGATION
################################################
df_credit_record_aggregated_1 = df_credit_record.loc[:, ['id', 'status']].groupby(['id', 'status']).value_counts().reset_index()
df_credit_record_aggregated_1.columns = ['id', 'status', 'count_status']

df_credit_record_aggregated_2 = df_credit_record.loc[:, ['id']].groupby(['id']).value_counts().reset_index()
df_credit_record_aggregated_2.columns = ['id', 'count_all']

df_credit_record_aggregated = df_credit_record_aggregated_1.merge(
    df_credit_record_aggregated_2, left_on=['id'], right_on=['id'])
df_credit_record_aggregated['share_status'] = df_credit_record_aggregated['count_status'] / df_credit_record_aggregated['count_all']
df_credit_record_aggregated = df_credit_record_aggregated.drop('count_status', axis=1)
del(df_credit_record_aggregated_1, df_credit_record_aggregated_2)

################################################
# CREDIT RECORD DATA - LONG TO WIDE
################################################
df_credit_record_wide = pd.pivot(df_credit_record_aggregated, index=['id', 'count_all'],
                                 columns='status', values='share_status')
df_credit_record_wide = df_credit_record_wide.fillna(value=0)
df_credit_record_wide = df_credit_record_wide.reset_index()
df_credit_record_wide['count_informative_statuses'] = df_credit_record_wide.count_all * (
    df_credit_record_wide['0'] + df_credit_record_wide['1'] +
    df_credit_record_wide['2'] + df_credit_record_wide['3'] +
    df_credit_record_wide['4'] + df_credit_record_wide['5'] +
    df_credit_record_wide['C'])
df_credit_record_wide = df_credit_record_wide.set_index(['id', 'count_all', 'count_informative_statuses'])

df_credit_record_wide.columns = ['status_' + col for col in df_credit_record_wide.columns]

del(df_credit_record_aggregated)

################################################
# CREDIT RECORD DATA - CUSTOMER CLASSIFICATION
################################################
df_credit_record_wide['customer_status'] = np.where(
    df_credit_record_wide.reset_index()["count_all"] < 12, 'too short history', np.where(
        df_credit_record_wide.reset_index()["count_informative_statuses"] < 12, 'too short history', np.where(
            df_credit_record_wide['status_X'] == 1, 'too short history', np.where(
                df_credit_record_wide['status_C'] == 1, 'perfect customer', np.where(
                    df_credit_record_wide['status_C'] + df_credit_record_wide['status_X'] == 1, 'perfect customer', np.where(
                        df_credit_record_wide['status_5'] > 0, 'bad customer', np.where(
                            df_credit_record_wide['status_4'] > 0, 'bad customer', np.where(
                                df_credit_record_wide['status_C'] + df_credit_record_wide['status_X'] + df_credit_record_wide['status_0'] > 0.7, 'good customer', np.where(
                                    df_credit_record_wide['status_C'] + df_credit_record_wide['status_X'] + df_credit_record_wide['status_0'] + df_credit_record_wide['status_1'] > 0.7, 'moderate customer', np.where(
                                        (df_credit_record_wide['status_3'] < 0.2) & (df_credit_record_wide['status_4'] == 0) & (df_credit_record_wide['status_5'] == 0), 'moderate customer', 'other'
))))))))))

df_credit_record_wide = df_credit_record_wide.reset_index().loc[:, ['id', 'customer_status']]

################################################
# MERGING CREDIT RECORD DATA WITH APPLICATION DATA
################################################
df_merged = df_application_record.merge(df_credit_record_wide,
                                        left_on=['id'], right_on=['id'])
del(df_application_record, df_credit_record_wide)

################################################
# DATA SELECTION
################################################
df_merged = df_merged.loc[~(df_merged.customer_status == 'too short history')]
df_merged = df_merged.set_index('id')
df_merged.occupation_type = df_merged.occupation_type.fillna("Unknown")

################################################
# SAVE DATA
################################################
df_merged.to_pickle('data/processed/df_application_record_classified_raw.pickle')
