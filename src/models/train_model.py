import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from src.utils.utils import read_pickle_data, save_model, save_train_test_data
from src.models.udfs_modelling import summarise_metrices, get_confusion_matrix
from src.models.udfs_modelling import reclassify_by_treshold
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.items():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def add_pca_components(X, numeric_cols_to_scale, nb_pca_components):
    X_pca = X.copy()
    cat_features = X.select_dtypes("object_").columns

    pca_num_proc = make_pipeline(StandardScaler())
    pca_cat_proc = make_pipeline(MultiColumnLabelEncoder())

    pca_preprocessor = make_column_transformer(
        # 'make_column_transformer' function to be able to
        # use custom class 'MultiColumnLabelEncoder'
        (pca_num_proc, numeric_cols_to_scale),
        (pca_cat_proc, cat_features)
    )

    pca_pipeline = make_pipeline(
        pca_preprocessor,
        PCA(n_components=nb_pca_components)
    )
    X_components = pca_pipeline.fit_transform(X_pca)

    component_names = [f"PC{i+1}" for i in range(X_components.shape[1])]
    X_components = pd.DataFrame(X_components, columns=component_names)
    X = pd.concat([X, X_components.set_index(X.index)], axis=1)

    return(X)


def data_preprocessing(X_train, X_test, one_hot_min_frequency,
                       numeric_cols_to_scale, nb_pca_components):
    pca_cols = list(range(nb_pca_components))
    pca_cols = ['PC' + str(num + 1) for num in pca_cols]
    numeric_cols_to_scale.extend(pca_cols)
    cat_features = X_train.select_dtypes("object_").columns

    num_proc = Pipeline(steps=[
        ('stand_scale', StandardScaler())
    ])
    cat_proc = Pipeline(steps=[
        ('one_hot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                  min_frequency=one_hot_min_frequency))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', num_proc, numeric_cols_to_scale),
        ('categorical', cat_proc, cat_features)
    ])

    preproc_pipeline = Pipeline(steps=[
        ('features_preproc', preprocessor)
    ])
    preproc_pipeline.fit(X_train)

    X_train_preproc = preproc_pipeline.transform(X_train)
    X_test_preproc = preproc_pipeline.transform(X_test)
    return X_train_preproc, X_test_preproc


def data_sampling(X_train, y_train, under_sampler_strategy,
                  over_sampler_strategy, tomek_sampler_strategy, seed):
    sampling_pipeline = Pipeline(steps=[
        ('under', RandomUnderSampler(sampling_strategy=under_sampler_strategy)),
        ('over', RandomOverSampler(random_state=seed,
                                   sampling_strategy=over_sampler_strategy)),
        ('tomek', TomekLinks(sampling_strategy=tomek_sampler_strategy))
    ])

    X_train_sampl, y_train_sampl = sampling_pipeline.fit_resample(X_train,
                                                                  y_train)
    return X_train_sampl, y_train_sampl


def fit_rf_model(X_train, y_train, params_rf):
    model = RandomForestClassifier(**params_rf)
    model = model.fit(X_train, y_train)
    return model


def model_training_main(data_processed_location,
                        gathered_feat_eng_data_filename,
                        target_name, numeric_cols_to_scale, nb_pca_components,
                        test_size, one_hot_min_frequency,
                        under_sampler_strategy, over_sampler_strategy,
                        tomek_sampler_strategy,
                        train_test_data_filename,
                        params_rf, model_location, model_filename,
                        prediction_threshold, seed):

    print("LOG PARAMS")
    param_keys = list(locals().keys())
    param_values = list(locals().values())
    for i in list(range(len(param_keys))):
        print(param_keys[i] + ": " + str(param_values[i]))

    print("LOAD DATA")
    df_merged = read_pickle_data(data_processed_location,
                                 gathered_feat_eng_data_filename)
    y = df_merged[target_name]
    X = df_merged.drop(target_name, axis=1)
    del(df_merged)

    print("PCA PIPELINE")
    X = add_pca_components(X, numeric_cols_to_scale, nb_pca_components)

    print("TRAIN TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=test_size,
                                                        random_state=seed)

    print("TRAIN & TEST DATA PREPROCESSING")
    X_train, X_test = data_preprocessing(X_train, X_test,
                                         one_hot_min_frequency,
                                         numeric_cols_to_scale,
                                         nb_pca_components)

    print("TRAIN DATA SAMPLING")
    X_train, y_train = data_sampling(X_train, y_train, under_sampler_strategy,
                                     over_sampler_strategy,
                                     tomek_sampler_strategy, seed)

    with mlflow.start_run():
        print("MODEL FITTING")
        rf_model = fit_rf_model(X_train, y_train, params_rf)

        print("DATA SAVING")
        save_train_test_data(X_train, y_train, X_test, y_test,
                             data_processed_location, train_test_data_filename)

        print("MODEL SAVING")
        save_model(rf_model, model_location, model_filename)

        print("MODEL PERFORMANCE")
        y_test_predictions = rf_model.predict_proba(X_test)
        y_test_predictions = reclassify_by_treshold(y_test_predictions,
                                                    prediction_threshold)
        average_precision, f1, precision, recall = summarise_metrices(
            y_test, y_test_predictions)
        _ = get_confusion_matrix(y_test, y_test_predictions)

        print("LOG PARAMS AND METRICS TO MLFLOW")
        for i in list(range(len(param_keys))):
            if (not(param_keys[i] == 'params_rf')):
                mlflow.log_param(param_keys[i], param_values[i])
        mlflow.log_param("class_weight", params_rf["class_weight"])
        mlflow.log_param("criterion", params_rf["criterion"])
        mlflow.log_param("max_depth", params_rf["max_depth"])
        mlflow.log_param("max_features", params_rf["max_features"])
        mlflow.log_param("n_estimators", params_rf["n_estimators"])

        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(rf_model, "model",
                                     registered_model_name="RF_Classification")
        else:
            mlflow.sklearn.log_model(rf_model, "model")


if __name__ == "__main__":
    seed = 12345
    data_processed_location = 'data/processed/'
    model_location = 'models/'
    gathered_feat_eng_data_filename = 'df_gathered_post_feature_eng.pickle'
    train_test_data_filename = 'df_train_test_rf'
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
    model_filename = 'latest_rf_model.sav'
    params_rf = {
        'class_weight': 'balanced',
        'criterion': 'gini',
        'max_depth': 12,
        'max_features': None,
        'n_estimators': 137
    }
    prediction_threshold = 0.84

    print("##### MODEL TRAINING #####")
    model_training_main(data_processed_location,
                        gathered_feat_eng_data_filename,
                        target_name, numeric_cols_to_scale, nb_pca_components,
                        test_size, one_hot_min_frequency,
                        under_sampler_strategy, over_sampler_strategy,
                        tomek_sampler_strategy,
                        train_test_data_filename,
                        params_rf, model_location, model_filename,
                        prediction_threshold, seed)
