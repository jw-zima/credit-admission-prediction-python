import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, average_precision_score


def get_summary(df, threshold):
    """Extract precision, recall and fscore for given threshold
    Args:
    df (data.frame): dataset with precision, recall and f1 for all thresholds
    threshold (numeric): cut off threshold for predition
    Yields:
    data.frame: Single line with stast for selected threshold
    Examples:
    >>> get_summary(tbl_results_pr, 0.5)
    """
    return df.loc[abs(df.thresholds - threshold) ==
                  min(abs(df.thresholds - threshold)), :]


def reclassify_by_treshold(predictions_proba, treshold):
    """Classify as 1 observations above a given threshold, as 0 otherwise
    Args:
    predictions_proba (series): series with prediction pseudo-probabilities
    threshold (numeric): cut off threshold for predition
    Yields:
    series: Binary predictions
    Examples:
    >>> reclassify_by_treshold(test_predictions_proba_final, 0.75)
    """
    return predictions_proba[:, 1] > treshold


def compute_f1(precision, recall):
    """Compute f1 score given precision and recall
    Args:
    precision (numeric): precision
    recall (numeric): recall
    Yields:
    numeric: f1 score
    Examples:
    >>> compute_f1(0.2, 0.016)
    """
    return (2 * precision * recall) / (precision + recall)


def summarise_metrices(y_test, y_hat):
    """Computes average_precision, f1, precision and recall given
    predictions and actual values
    Args:
    y_test (series): series with predicted classes
    y_hat (series): series with actual classes
    Yields:
    Printed metrices
    Examples:
    >>> summarise_metrices(y_test = y_test, y_hat = test_predictions_lgbm)
    """
    print("average_precision test: " +
          str(round(average_precision_score(y_test, y_hat), 2)))
    print("f1 test: " + str(round(f1_score(y_test, y_hat), 2)))
    print("precision test: " + str(round(precision_score(y_test, y_hat), 2)))
    print("recall test: " + str(round(recall_score(y_test, y_hat), 2)))


def get_confusion_matrix(y_test, y_hat):
    """Computes and renames rows & columns of the confusion matrix
    Args:
    y_test (series): series with predicted classes
    y_hat (series): series with actual classes
    Yields:
    data.frame: Confusion matrix
    Examples:
    >>> get_confusion_matrix(y_test, y_hat)
    """
    cm = confusion_matrix(y_test, y_hat)
    cm = pd.DataFrame(data=cm,
                      columns=['Actual Negative:0', 'Actual Positive:1'],
                      index=['Predict Negative:0', 'Predict Positive:1'])
    return cm


def format_confusion_matrix(y_test, y_hat):
    """Generate plot with conditioanlly formatted confusion matrix
    Args:
    y_test (series): series with predicted classes
    y_hat (series): series with actual classes
    Yields:
    sns plot: Confusion matrix with applied ocnditional formatting
    Examples:
    >>> format_confusion_matrix(y_test, y_hat)
    """
    cm_matrix = get_confusion_matrix(y_test, y_hat)
    return sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


def get_rf_feature_imp(rf_model):
    """Extraxt and plot model features sorted by their importance
    Args:
    rf_model (model): random forest model
    Yields:
    plot: Bar plot with variables impotrance
    Examples:
    >>> get_rf_feature_imp(final_rf_model)
    """
    vars = rf_model.feature_names_in_.tolist()
    imp = rf_model.feature_importances_.tolist()
    rf_feature_importance = pd.DataFrame({'variable': vars,
                                          'importance': imp})
    rf_feature_importance = rf_feature_importance.sort_values('importance',
                                                              ascending=False)

    print(ggplot(data=rf_feature_importance)
          + geom_col(aes(x='variable', y='importance'))
          + theme_light()
          + xlab("variable")
          + ylab("importance")
          + scale_x_discrete(limits=rf_feature_importance.variable)
          + theme(axis_text_x=element_text(angle=90))
          )


def extract_and_plot_hyperopt_trials(trials):
    """Extraxt and plot loss in all hyperopt trials for each optimized param
    Args:
    trials_lgbm (Trials): hyperopt trials
    Yields:
    plot: Plot showing loss in all hyperopt trials for each optimized param
    Examples:
    >>> extract_and_plot_hyperopt_trials(trials_lgbm)
    """
    results = pd.DataFrame()
    for val in trials.vals:
        results[val] = trials.vals[val]
    results['loss'] = [x['loss'] for x in trials.results]

    for col in results.columns[:-1]:
        print(ggplot(aes(x=results[col], y=results['loss']))
              + geom_point()
              + theme_light())


def save_intermin_data_and_model(X_train_data, y_train_data, X_test_data,
                                 y_test_data, classifier, model_name):
    """Saving data used for given model along with saving the model itself
    Args:
    X_train_data (data.frame): dataset used to train model
    y_train_data (series): target variable used to train model
    X_test_data (data.frame): dataset used to test model
    y_test_data (series): target variable for model testing
    classifier (object): fitted model
    model_name (string): name of the model that would be added to the name
    of the data dump
    Yields:
    plot: Saves data used for given model along with saving the model itself
    Examples:
    >>> save_intermin_data_and_model(X_train_data, y_train_data, X_test_data,
                                     y_test_data, lgbm_final, 'lgbm')
    """
    data_dict = {'X_train_data': X_train_data, 'y_train_data': y_train_data,
                 'X_test_data': X_test_data, 'y_test_data': y_test_data,
                 'classifier': classifier}
    outfile = open('data/interim/model_training_' + model_name, 'wb')
    pickle.dump(data_dict, outfile)
    outfile.close()


def get_stats_by_gender(X, y, model, group_one, with_gender=True):
    """Check 3 model fairness criteria: Demographic parity / statistical parity,
    Equal opportunity, Equal accuracy
    Args:
    X (data.frame): dataset used to train model
    y (series): target variable used to train model
    group_one (series): flag which observations belong to firth group
    with_gender (boolean): info if model used gender column
    Yields:
    Computes confusion matrices for both geners, share of both class,
    admission rate for each group, computes accuracy, recall.
    Examples:
    >>> get_stats_by_gender(X_test_data, y_test_data,
                            classifier, X_test_data["code_gender"]==1)
    """
    if with_gender is not True:
        X = X.drop('code_gender', axis=1)

    preds = model.predict(X)
    y_zero = y[group_one==False]
    preds_zero = preds[group_one==False]
    y_one = y[group_one]
    preds_one = preds[group_one]

    print("1. Demographic parity / statistical parity")
    print("\nTotal observations:", len(preds))
    pct_zero = round(len(preds_zero)/len(preds)*100, 2)
    pct_one = round(len(preds_one)/len(preds)*100, 2)
    print("Group 0:", len(preds_zero), "({}%)".format(pct_zero))
    print("Group 1:", len(preds_one), "({}%)".format(pct_one))

    print("\nTotal approvals:", preds.sum())
    approvals_zero_to_all = round(preds_zero.sum()/sum(preds)*100, 2)
    approvals_one_to_all = round(preds_one.sum()/sum(preds)*100, 2)
    approvals_zero_to_females = round(preds_zero.sum()/len(preds_zero)*100, 2)
    approvals_one_to_males = round(preds_one.sum()/len(preds_one)*100, 2)
    print("Group 0:", preds_zero.sum(),
          "({}% of all approvals)".format(approvals_zero_to_all),
          "({}% of female approvals)".format(approvals_zero_to_females)
          )
    print("Group 1:", preds_one.sum(),
          "({}% of approvals)".format(approvals_one_to_all),
          "({}% of female approvals)".format(approvals_one_to_males)
          )

    print("\n2. Equal opportunity")
    cm_zero = get_confusion_matrix(y_zero, preds_zero)
    cm_one = get_confusion_matrix(y_one, preds_one)

    sns.set(rc={'figure.figsize': (20, 8)})
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(cm_zero, annot=True, fmt='d', cmap='YlGnBu', ax=ax[0])
    ax[0].set_title('Group 0')
    sns.heatmap(cm_one, annot=True, fmt='d', cmap='YlGnBu', ax=ax[1])
    ax[1].set_title('Group 1')
    fig.show()

    print("\nSensitivity / Recall:")
    recall_zero = round(cm_zero.iloc[1, 1] /
                        (cm_zero.iloc[0, 1] + cm_zero.iloc[1, 1]) * 100, 2)
    recall_one = round(cm_one.iloc[1, 1] /
                       (cm_one.iloc[0, 1] + cm_one.iloc[1, 1]) * 100, 2)
    print("Group 0: {}%".format(recall_zero))
    print("Group 1: {}%".format(recall_one))

    print("\n3. Equal accuracy")
    print("\nOverall accuracy: {}%".
          format(round((preds == y).sum()/len(y)*100, 2)))
    print("Group 0: {}%".
          format(round((preds_zero == y_zero).sum()/len(y_zero)*100, 2)))
    print("Group 1: {}%".
          format(round((preds_one == y_one).sum()/len(y_one)*100, 2)))
