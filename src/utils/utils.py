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


if __name__ == "__main__":
    print("hello")
