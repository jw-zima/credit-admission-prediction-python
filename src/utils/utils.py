import pandas as pd
import pickle


def lower_cols(df):
    df.columns = df.columns.str.lower()
    return df


def read_csv_data(folder_location, data_filename):
    df = pd.read_csv(folder_location + data_filename)
    df = lower_cols(df)
    return df


def save_pickle_data(df, folder_location, data_filename):
    df.to_pickle(folder_location + data_filename)


def read_pickle_data(folder_location, data_filename):
    df = pd.read_pickle(folder_location + data_filename)
    return df


def save_model(model, folder_location, model_filename):
    pickle.dump(model, open(folder_location + model_filename, 'wb'))


def save_train_test_data(X_train_data, y_train_data, X_test_data, y_test_data,
                         folder_location, data_filename):
    """Saving train and test data used to fit and test given model
    Args:
    X_train_data (data.frame): dataset used to train model
    y_train_data (series): target variable used to train model
    X_test_data (data.frame): dataset used to test model
    y_test_data (series): target variable for model testing
    Yields:
    saved file: Saves train and test data
    Examples:
    >>> save_train_test_data(X_train_data, y_train_data,
                             X_test_data, y_test_data,
                             data_processed_location, train_test_data_filename)
    """
    data_dict = {'X_train_data': X_train_data, 'y_train_data': y_train_data,
                 'X_test_data': X_test_data, 'y_test_data': y_test_data}
    outfile = open(folder_location + data_filename, 'wb')
    pickle.dump(data_dict, outfile)
    outfile.close()


def read_train_test_data(folder_location, data_filename):
    infile = open(folder_location + data_filename, 'rb')
    loaded_dict = pickle.load(infile)
    infile.close()
    return loaded_dict


if __name__ == "__main__":
    print("hello")
