import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pylab as plt

mpl.style.use('seaborn-paper')

from utils.constants import max_seq_len, nb_classes


def load_dataset_at(index, method, proto_num, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    dim = 1

    train_data1 = "data/all-raw-train-data-%s-%s-%s.txt" % (index, method, str(proto_num))
    test_data1 = "data/all-raw-test-data-%s-%s-%s.txt" % (index, method, str(proto_num))

    train_data2 = "data/all-dtw_features-train-data-%s-%s-%s.txt" % (index, method, str(proto_num))
    test_data2 = "data/all-dtw_features-test-data-%s-%s-%s.txt" % (index, method, str(proto_num))

    train_labels = "data/all-train-label-%s-%s-%s.txt" % (index, method, str(proto_num))
    test_labels = "data/all-test-label-%s-%s-%s.txt" % (index, method, str(proto_num))


    is_timeseries = True # assume all input data is univariate time series

    if os.path.exists(train_data1):
        df = pd.read_csv(train_data1, delimiter=' ', header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (train_data1))
    X_train1 = df.values
    X_train1 = np.reshape(X_train1, (np.shape(X_train1)[0], dim, int(np.shape(X_train1)[1]/(dim))))

    if os.path.exists(train_data2):
        df = pd.read_csv(train_data2, delimiter=' ', header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (train_data2))

    X_train2 = df.values
    X_train2 = np.reshape(X_train2, (np.shape(X_train2)[0], proto_num, int(np.shape(X_train2)[1]/(proto_num))))

    if normalize_timeseries:
        X_train_min = np.min(X_train2)
        X_train_max = np.max(X_train2)
        X_train2 = 2. * (X_train2 - X_train_min) / (X_train_max - X_train_min) - 1.

    if os.path.exists(train_labels):
        df = pd.read_csv(train_labels, delimiter=' ', header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (train_labels))

    y_train = df[[1]].values

    no_classes = nb_classes(index) #len(np.unique(y_train))


    if os.path.exists(test_data1):
        df = pd.read_csv(test_data1, delimiter=' ', header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (test_data1))
    X_test1 = df.values
    X_test1 = np.reshape(X_test1, (np.shape(X_test1)[0], dim, int(np.shape(X_test1)[1]/(dim))))

    if os.path.exists(test_data2):
        df = pd.read_csv(test_data2, delimiter=' ', header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (test_data2))

    X_test2 = df.values
    X_test2 = np.reshape(X_test2, (np.shape(X_test2)[0], proto_num, int(np.shape(X_test2)[1]/(proto_num))))

    if normalize_timeseries:
        X_test2 = 2. * (X_test2 - X_train_min) / (X_train_max - X_train_min) - 1.

    if os.path.exists(test_labels):
        df = pd.read_csv(test_labels, delimiter=' ', header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (test_labels))

    y_test = df[[1]].values

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train1.shape[0], "Number of test samples : ", X_test1.shape[0])
        print("Number of classes : ", no_classes)
        print("Sequence length : ", X_train1.shape[-1])


    return X_train1, X_train2, y_train, X_test1, X_test2, y_test, is_timeseries


if __name__ == "__main__":
    # word_list = []
    # seq_len_list = []
    # classes = []
    #
    # for index in range(6, 9):
    #     x, y, x_test, y_test, is_timeseries = load_dataset_at(index)
    #     nb_words, seq_len = calculate_dataset_metrics(x)
    #     print("-" * 80)
    #     print("Dataset : ", index + 1)
    #     print("Train :: X shape : ", x.shape, "Y shape : ", y.shape, "Nb classes : ", len(np.unique(y)))
    #     print("Test :: X shape : ", x_test.shape, "Y shape : ", y_test.shape, "Nb classes : ", len(np.unique(y)))
    #     print("Classes : ", np.unique(y))
    #     print()
    #
    #     word_list.append(nb_words)
    #     seq_len_list.append(seq_len)
    #     classes.append(len(np.unique(y)))
    #
    # print("Word List : ", word_list)
    # print("Sequence length list : ", seq_len_list)
    # print("Max number of classes : ", classes)

    #print()
    plot_dataset(dataset_id=39, seed=1, limit=1, cutoff=None, normalize_timeseries=False,
                 plot_classwise=True)
