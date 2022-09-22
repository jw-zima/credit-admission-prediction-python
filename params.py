import datetime


execution_start = datetime.datetime.now()
seed = 12345
data_raw_location = 'data/raw/'
data_processed_location = 'data/processed/'
model_location = 'models/'
raw_application_filename = 'application_record.csv'
raw_credit_hist_filename = 'credit_record.csv'
gathered_data_filename = 'df_gathered.pickle'
gathered_feat_eng_data_filename = 'df_gathered_post_feature_eng.pickle'
processed_data_filename = 'df_gathered.pickle'
target_name = 'target'
numeric_cols_to_scale = ['cnt_children', 'amt_income_total',
                         'cnt_fam_members', 'amt_income_per_person',
                         'age', 'job_tenure']
nb_pca_components = 3
test_size = 0.2
one_hot_min_frequency = 30
under_sampler_strategy = 0.1
over_sampler_strategy = 0.9
tomek_sampler_strategy = 'majority'
model_filename = 'rf_model_' + execution_start.strftime("%Y-%m-%d_%H-%M-%S") \
    + '.sav'
params_rf = {
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': 12,
    'max_features': None,
    'n_estimators': 137
}
