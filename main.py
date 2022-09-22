from src.data.data_gathering import data_gathering_main
from src.features.build_features import features_engineering_main
from src.models.train_model import model_training_main
import params

if __name__ == "__main__":
    print(params.data_raw_location)
    print(params.seed)

    print("##### DATA GATHERING #####")
    data_gathering_main(params.data_raw_location,
                        params.data_processed_location,
                        params.raw_application_filename,
                        params.raw_credit_hist_filename,
                        params.processed_data_filename)

    print("##### FEATURES ENGINEERING AND SELECTION #####")
    features_engineering_main(params.data_processed_location,
                              params.gathered_data_filename,
                              params.gathered_feat_eng_data_filename)

    print("##### MODEL TRAINING #####")
    model_training_main(params.data_processed_location,
                        params.gathered_feat_eng_data_filename,
                        params.target_name, params.numeric_cols_to_scale,
                        params.nb_pca_components,
                        params.test_size, params.one_hot_min_frequency,
                        params.under_sampler_strategy,
                        params.over_sampler_strategy,
                        params.tomek_sampler_strategy,
                        params.params_rf,
                        params.model_location, params.model_filename,
                        params.seed)

    print("##### DONE #####")
