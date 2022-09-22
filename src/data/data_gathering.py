import pandas as pd
import numpy as np
from src.utils.utils import read_csv_data, save_pickle_data


def aggregate_credit_record(df):
    df_aggregated_1 = df.loc[:, ['id', 'status']]. \
        groupby(['id', 'status']).value_counts().reset_index()
    df_aggregated_1.columns = ['id', 'status', 'count_status']

    df_aggregated_2 = df.loc[:, ['id']]. \
        groupby(['id']).value_counts().reset_index()
    df_aggregated_2.columns = ['id', 'count_all']

    df_aggregated = df_aggregated_1. \
        merge(df_aggregated_2, left_on=['id'], right_on=['id'])
    df_aggregated['share_status'] = df_aggregated['count_status'] \
        / df_aggregated['count_all']
    df_aggregated = df_aggregated.drop('count_status', axis=1)
    return df_aggregated


def spread_credit_record(df):
    df_wide = pd.pivot(df, index=['id', 'count_all'],
                       columns='status', values='share_status')
    df_wide = df_wide.fillna(value=0)
    df_wide = df_wide.reset_index()
    df_wide['count_informative_statuses'] = df_wide.count_all * (
        df_wide['0'] + df_wide['1'] + df_wide['2'] + df_wide['3'] +
        df_wide['4'] + df_wide['5'] + df_wide['C'])
    df_wide = df_wide. \
        set_index(['id', 'count_all', 'count_informative_statuses'])

    df_wide.columns = ['status_' + col for col in df_wide.columns]
    return df_wide


def create_customer_status(df):
    df['customer_status'] = np.where(
        df.reset_index()["count_all"] < 12, 'too short history', np.where(
            df.reset_index()["count_informative_statuses"] < 12, 'too short history', np.where(
                df['status_X'] == 1, 'too short history', np.where(
                    df['status_C'] == 1, 'perfect customer', np.where(
                        df['status_C'] + df['status_X'] == 1, 'perfect customer', np.where(
                            df['status_5'] > 0, 'bad customer', np.where(
                                df['status_4'] > 0, 'bad customer', np.where(
                                    df['status_C'] + df['status_X'] + df['status_0'] > 0.7, 'good customer', np.where(
                                        df['status_C'] + df['status_X'] + df['status_0'] + df['status_1'] > 0.7, 'moderate customer', np.where(
                                            (df['status_3'] < 0.2) & (df['status_4'] == 0) & (df['status_5'] == 0), 'moderate customer', 'other'
                                            ))))))))))

    df = df.reset_index().loc[:, ['id', 'customer_status']]
    return df


def merge_application_and_credit_data(application_data, credit_data):
    df_merged = application_data.\
        merge(credit_data, left_on=['id'], right_on=['id'])
    return df_merged


def filter_data_for_training(df):
    df = df.loc[~(df.customer_status == 'too short history')]
    df = df.set_index('id')
    df.occupation_type = df.occupation_type.fillna("Unknown")
    return df


def data_gathering_main(data_raw_location, data_processed_location,
                        raw_application_filename, raw_credit_hist_filename,
                        processed_data_filename):
    print("LOAD DATA")
    df_application_record = read_csv_data(data_raw_location, raw_application_filename)
    df_credit_record = read_csv_data(data_raw_location, raw_credit_hist_filename)

    print("CREDIT RECORD DATA - AGGREGATION")
    df_credit_record_aggregated = aggregate_credit_record(df_credit_record)
    del(df_credit_record)

    print("CREDIT RECORD DATA - LONG TO WIDE")
    df_credit_record_wide = spread_credit_record(df_credit_record_aggregated)
    del(df_credit_record_aggregated)

    print("CREDIT RECORD DATA - CUSTOMER CLASSIFICATION")
    df_credit_record_wide = create_customer_status(df_credit_record_wide)

    print("MERGE CREDIT RECORD DATA WITH APPLICATION DATA")
    df_merged = merge_application_and_credit_data(df_application_record,
                                                  df_credit_record_wide)
    del(df_application_record, df_credit_record_wide)

    print("DATA SELECTION")
    df_merged = filter_data_for_training(df_merged)

    print("DATA SAVING")
    save_pickle_data(df_merged, data_processed_location, processed_data_filename)


if __name__ == "__main__":
    data_raw_location = 'data/raw/'
    data_processed_location = 'data/processed/'
    raw_application_filename = 'application_record.csv'
    raw_credit_hist_filename = 'credit_record.csv'
    processed_data_filename = 'df_gathered.pickle'

    print("##### DATA GATHERING #####")
    data_gathering_main(data_raw_location, data_processed_location,
                        raw_application_filename, raw_credit_hist_filename,
                        processed_data_filename)
