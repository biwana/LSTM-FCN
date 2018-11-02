import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

mpl.style.use('seaborn-paper')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.models import Model
from keras.layers import Permute
from keras.optimizers import Adam, SGD, Nadam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

from utils.generic_utils import load_dataset_at
from utils.constants import max_seq_len, nb_classes


def train_model(model:Model, dataset_id, method, proto_num, dataset_prefix, nb_iterations=100000, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, opt='Adam', learning_rate=1e-3, early_stop=False, balance_classes=True, run_ver=''):
    X_train1, X_train2, y_train, X_test1, X_test2, y_test, is_timeseries = load_dataset_at(dataset_id, method, proto_num, normalize_timeseries=normalize_timeseries)

    #calculate num of batches
    nb_epochs = math.ceil(nb_iterations * (batch_size / X_train1.shape[0]))

    if balance_classes == True:
        classes = np.arange(0, nb_classes(dataset_id)) #np.unique(y_train)
        le = LabelEncoder()
        y_ind = le.fit_transform(y_train.ravel())
        recip_freq = len(y_train) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
        class_weight = recip_freq[le.transform(classes)]

        print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, nb_classes(dataset_id))
    y_test = to_categorical(y_test, nb_classes(dataset_id))

    #not used
    factor = 1. / np.cbrt(2)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=math.ceil(nb_epochs / 20), mode='auto',
                                  factor=factor, cooldown=0, min_lr=learning_rate/10., verbose=2)

    model_checkpoint1 = ModelCheckpoint("./weights/%s_%s_%s_%sloss_weights.h5" % (dataset_prefix, method, str(proto_num), run_ver), verbose=2,
                                       monitor='loss', save_best_only=True, save_weights_only=True)
    model_checkpoint2 = ModelCheckpoint("./weights/%s_%s_%s_%sval_acc_weights.h5" % (dataset_prefix, method, str(proto_num), run_ver), verbose=2,
                                       monitor='val_acc', save_best_only=True, save_weights_only=True)

    tensorboard = TensorBoard(log_dir='./logs/%s%s_%s_%s' % (run_ver, dataset_prefix, method, str(proto_num)), batch_size=batch_size)
    csv_logger = CSVLogger('./logs/%s%s_%s_%s.csv' % (run_ver, dataset_prefix, method, str(proto_num)))
    if early_stop:
        early_stopping = EarlyStopping(monitor='loss', patience=500, mode='auto', verbose=2, restore_best_weights=True)
        callback_list = [model_checkpoint1, model_checkpoint2, early_stopping, tensorboard, csv_logger]
    else:
        callback_list = [model_checkpoint1, model_checkpoint2, tensorboard, csv_logger]

    if opt == 'SGD':
        optm = SGD(lr=learning_rate, momentum=0.9, decay=5e-4)
    elif opt == 'Nadam':
        optm = Nadam(lr=learning_rate)
    elif opt == 'Adam_decay':
        optm = Adam(lr=learning_rate, decay=9./nb_iterations)
    else:
        optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if balance_classes:
        model.fit([X_train1, X_train2], y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=([X_test1, X_test2], y_test))
    else:
        model.fit([X_train1, X_train2], y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callback_list, verbose=2, validation_data=([X_test1, X_test2], y_test))


def evaluate_model(model:Model, dataset_id, method, proto_num, dataset_prefix, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False, checkpoint_prefix="loss"):
    X_train1, X_train2, y_train, X_test1, X_test2, y_test, is_timeseries = load_dataset_at(dataset_id, method, proto_num, normalize_timeseries=normalize_timeseries)
    
    y_test = to_categorical(y_test, nb_classes(dataset_id))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("./weights/%s_%s_%s_%s_weights.h5" % (dataset_prefix, method, str(proto_num), checkpoint_prefix))


    print("\nEvaluating : ")
    loss, accuracy = model.evaluate([X_test1, X_test2], y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)

    return accuracy


