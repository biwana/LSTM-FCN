
from keras.models import Model
from keras.layers import Input, Dense, multiply, concatenate, Activation, Lambda
from keras.layers import PReLU, LSTM
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import MaxPooling1D, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

from utils.constants import max_seq_len, nb_classes
from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
                                cutoff_sequence, plot_dataset
import sys
import math
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"


TRAINABLE = True

def slice_seq(x):
    return x[:, :1]

def slice_dtw(x):
    return x[:, 1:]

def play_model(nb_cnn, proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x = Permute((2, 1))(ip1)

    for i in range(nb_cnn):
        factor = i if i < 3 else 3
        nb_nodes = 64 * 2 ** i

        x = Conv1D(nb_nodes, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
        x = Conv1D(nb_nodes, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
        if i > 2:
            x = Conv1D(nb_nodes, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)

        x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    return model

def train_model(model:Model, dataset_id, method, proto_num, dataset_prefix, nb_iterations=100000, batch_size=128, val_subset=None, cutoff=None, normalize_timeseries=False, learning_rate=1e-3, early_stop=False, balance_classes=True, run_ver=''):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id, method, proto_num, normalize_timeseries=normalize_timeseries)

    #calculate num of batches
    nb_epochs = math.ceil(nb_iterations * (batch_size / X_train.shape[0]))

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

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=math.ceil(nb_epochs / 20), mode='auto',
                                  factor=factor, cooldown=0, min_lr=learning_rate/10., verbose=2)

    if early_stop:
        early_stopping = EarlyStopping(monitor='loss', patience=500, mode='auto', verbose=2, restore_best_weights=True)
        callback_list = [early_stopping]
    else:
        callback_list = []

    optm = Adam(lr=learning_rate)
    #optm = SGD(lr=learning_rate, momentum=0.9, decay=5e-4)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]


    if balance_classes:
        model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))
    else:
        model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=callback_list, verbose=2, validation_data=(X_test, y_test))


if __name__ == "__main__":
    dataset = sys.argv[1]
    method = sys.argv[2]
    proto_num = int(sys.argv[3])

    max_seq_lenth = max_seq_len(dataset)
    nb_class = nb_classes(dataset)
    nb_cnn = int(round(math.log(max_seq_lenth, 2))-3)
    print("Number of Pooling Layers: %s" % str(nb_cnn))

    #model = lstm_fcn_model(proto_num, max_seq_lenth, nb_class)
    #model = alstm_fcn_model(proto_num, max_seq_lenth, nb_class)

    #model = cnn_raw_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_dtwfeatures_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_earlyfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_midfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_latefusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)

    #model = vgg_raw_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = vgg_dtwfeatures_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = vgg_earlyfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = vgg_midfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = vgg_latefusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)

    model = play_model(nb_cnn, proto_num, max_seq_lenth, nb_class)


    train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=50000, batch_size=50, learning_rate=0.0001, early_stop=False, balance_classes=False, run_ver='vgg_')
    #train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=28000, batch_size=64, learning_rate=0.001, early_stop=True)

    acc = evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="vgg_loss")
    np.savetxt("output/vgg/vgg-%s-%s-%s-loss-%s" % (dataset, method, str(proto_num), str(acc)), [acc])

    acc = evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="vgg_val_acc")
    np.savetxt("output/vgg/vgg-%s-%s-%s-vacc-%s" % (dataset, method, str(proto_num), str(acc)), [acc])
