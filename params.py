import datetime


execution_start = datetime.datetime.now()
seed = 12345
# files location
data_raw_location = 'data/raw/'
data_processed_location = 'data/processed/'
model_location = 'models/'
# data filenames
raw_application_filename = 'application_record.csv'
raw_credit_hist_filename = 'credit_record.csv'
gathered_data_filename = 'df_gathered.pickle'
gathered_feat_eng_data_filename = 'df_gathered_post_feature_eng.pickle'
processed_data_filename = 'df_gathered.pickle'
train_test_data_filename = 'df_train_test_rf'
# columns for modelling
target_name = 'target'
numeric_cols_to_scale = ['cnt_children', 'amt_income_total',
                         'cnt_fam_members', 'amt_income_per_person',
                         'age', 'job_tenure']
# feature engineering params
nb_pca_components = 3
one_hot_min_frequency = 30
under_sampler_strategy = 0.1
over_sampler_strategy = 0.9
tomek_sampler_strategy = 'majority'
# modelling params
test_size = 0.2
model_filename = 'rf_model_' + execution_start.strftime("%Y-%m-%d_%H-%M-%S") \
    + '.sav'
params_rf = {
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': 12,
    'max_features': None,
    'n_estimators': 1403
}
# prediction
prediction_threshold = 0.84
