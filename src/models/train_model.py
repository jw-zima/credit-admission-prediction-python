################################################
# IMPORT
################################################
import pandas as pd
import pickle
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


################################################
# PARAMS
################################################
seed = 12345
data_processed_location = 'data/processed/'
model_location = 'models/'
feat_eng_processed_data_filename = 'df_application_record_classified_post_feature_eng.pickle'
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


################################################
# MultiColumnLabelEncoder
################################################
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
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


################################################
# DATA LOAD
################################################
df_merged = pd.read_pickle(data_processed_location + feat_eng_processed_data_filename)

y = df_merged[target_name]
X = df_merged.drop(target_name, axis=1)

################################################
# PCA PIPELINE
################################################
X_pca = X.copy()
cat_features = X.select_dtypes("object_").columns
pca_num_cols_to_scale = ['cnt_children', 'amt_income_total', 'cnt_fam_members',
                         'amt_income_per_person', 'age', 'job_tenure']

pca_num_proc = make_pipeline(StandardScaler())
pca_cat_proc = make_pipeline(MultiColumnLabelEncoder())

pca_preprocessor = make_column_transformer(
    # 'make_column_transformer' function to be able to
    # use custom class 'MultiColumnLabelEncoder'
    (pca_num_proc, pca_num_cols_to_scale),
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

del(X_components, X_pca)

################################################
# DATA SPLIT
################################################
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=test_size,
                                                    random_state=seed)

################################################
# PREPROCESSING AND SAMPLING PIPELINES
################################################
pca_cols = list(range(nb_pca_components))
pca_cols = ['PC' + str(num + 1) for num in pca_cols]

num_cols_to_scale = pca_num_cols_to_scale.copy()
num_cols_to_scale.extend(pca_cols)

num_proc = Pipeline(steps=[
    ('stand_scale', StandardScaler())
])
cat_proc = Pipeline(steps=[
    ('one_hot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                              min_frequency=oh_min_frequency))
])

# preprocessing piepeine
preprocessor = ColumnTransformer(transformers=[
    ('numeric', num_proc, num_cols_to_scale),
    ('categorical', cat_proc, cat_features)
])

preproc_pipeline = Pipeline(steps=[
    ('features_preproc', preprocessor)
])
preproc_pipeline.fit(X_train)

X_train_preproc = preproc_pipeline.transform(X_train)
X_test_preproc = preproc_pipeline.transform(X_test)

# sampling pipeline
sampling_pipeline = Pipeline(steps=[
    ('under', RandomUnderSampler(sampling_strategy=under_sampler_strategy)),
    ('over', RandomOverSampler(random_state=seed,
                               sampling_strategy=over_sampler_strategy)),
    ('tomek', TomekLinks(sampling_strategy=tomek_sampler_strategy))
])

X_train_sampl, y_train_sampl = sampling_pipeline.fit_resample(X_train_preproc,
                                                              y_train)
del(X_train_preproc)

################################################
# MODEL FITTING
################################################
rf_model = RandomForestClassifier(**params_rf)
rf_model = rf_model.fit(X_train_sampl, y_train_sampl)


pickle.dump(rf_model, open(model_location + model_filename, 'wb'))

################################################
# MAIN PLACEHOLDER
################################################
if __name__ == "__main__":
    print("Hello, World!")
    # TO BE DONE
