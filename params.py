seed = 12345
data_raw_location = 'data/raw/'
data_processed_location = 'data/processed/'
model_location = 'models/'
raw_application_filename = 'application_record.csv'
raw_credit_hist_filename = 'credit_record.csv'
processed_data_filename = 'df_application_record_classified_post_feature_eng.pickle'
target_name = 'target'
nb_pca_components = 3
test_size = 0.2
oh_min_frequency = 30
under_sampler_strategy = 0.1
over_sampler_strategy = 0.9
tomek_sampler_strategy = 'majority'
model_filename = 'latest_rf_model.sav'
params_rf = {
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': 12,
    'max_features': None,
    'n_estimators': 137
}
