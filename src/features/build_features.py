import pandas as pd
import numpy as np
from src.utils.utils import save_pickle_data, read_pickle_data


def create_new_features(df):
    df['amt_income_per_person'] = df['amt_income_total'] / df['cnt_fam_members']
    df['age'] = -1 * df['days_birth'] / 365
    df['job_tenure'] = -1 * df['days_employed'] / 365
    df['job_tenure'] = np.where(df["days_employed"] > 0, -1, df['job_tenure'])
    df['flag_employed'] = np.where(df["days_employed"] > 0, 0, 1)
    df['code_gender'] = np.where(df['code_gender'] == 'M', 1, 0)
    df['flag_own_car'] = np.where(df['flag_own_car'] == 'Y', 1, 0)
    df['flag_own_realty'] = np.where(df['flag_own_realty'] == 'Y', 1, 0)
    df['single_adult'] = np.where(
        (df['name_family_status'] == 'Married') \
        | (df['name_family_status'] == 'Civil marriage'), 0, 1)
    df['target'] = np.where((df['customer_status'] == 'bad customer') \
        | (df['customer_status'] == 'moderate customer'), 1, 0)
    df = df.astype({'flag_own_car': int,
                    'flag_own_realty': int,
                    'cnt_children': int,
                    'cnt_fam_members': int})
    return df


def select_features(df):
    return df.drop(['flag_mobil', 'flag_work_phone', 'flag_phone',
                    'flag_email', 'days_birth', 'days_employed',
                    'name_family_status', 'customer_status'], axis=1)


def features_engineering_main(data_processed_location,
                              input_filename, output_filename):
    print("LOAD DATA")
    df_merged = read_pickle_data(data_processed_location, input_filename)

    print("FEATURE ENGINEERING")
    df_merged = create_new_features(df_merged)

    print("FEATURES SELECTION")
    df_merged = select_features(df_merged)

    print("SAVE DATA")
    save_pickle_data(df_merged, data_processed_location,
                     output_filename)


if __name__ == "__main__":
    data_processed_location = 'data/processed/'
    gathered_data_filename = 'df_gathered.pickle'
    gathered_feat_eng_data_filename = 'df_gathered_post_feature_eng.pickle'


    print("##### FEATURES ENGINEERING AND SELECTION #####")
    features_engineering_main(data_processed_location,
                              gathered_data_filename,
                              gathered_feat_eng_data_filename)
