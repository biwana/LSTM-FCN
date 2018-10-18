from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation, Lambda
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import max_seq_len, nb_classes
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualize_context_vector, visualize_cam
from utils.layer_utils import AttentionLSTM


import numpy as np
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


TRAINABLE = True

def slice_seq(x):
    return x[:, :1]

def slice_dtw(x):
    return x[:, 1:]

def generate_model(proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x1 = LSTM(128)(ip1)
    x1 = Dropout(0.8)(x1)

    y1 = Permute((2, 1))(ip1)
    y1 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = GlobalAveragePooling1D()(y1)

    x1 = concatenate([x1, y1])

    x2 = AttentionLSTM(128)(ip2)
    x2 = Dropout(0.8)(x2)

    y2 = Permute((2, 1))(ip2)
    y2 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = GlobalAveragePooling1D()(y2)

    x2 = concatenate([x2, y2])

    x = concatenate([x1, x2])

    x = Dense(1024, activation='relu')(x)

    x = Dense(1024, activation='relu')(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_2(proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x1 = AttentionLSTM(128)(ip1)
    x1 = Dropout(0.8)(x1)

    y1 = Permute((2, 1))(ip1)
    y1 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y1 = GlobalAveragePooling1D()(y1)

    x1 = concatenate([x1, y1])

    x2 = AttentionLSTM(128)(ip2)
    x2 = Dropout(0.8)(x2)

    y2 = Permute((2, 1))(ip2)
    y2 = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y2 = GlobalAveragePooling1D()(y2)

    x2 = concatenate([x2, y2])

    x = concatenate([x1, x2])

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == "__main__":
    dataset = sys.argv[1]
    method = sys.argv[2]
    proto_num = int(sys.argv[3])

    max_seq_lenth = max_seq_len(dataset)
    nb_class = nb_classes(dataset)

    model = generate_model_2(proto_num, max_seq_lenth, nb_class)

    train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=100000, batch_size=50, learning_rate=0.0001)

    evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="loss")

    # visualize_context_vector(model, DATASET_INDEX, dataset_prefix='swedish_leaf', visualize_sequence=True,
    #                          visualize_classwise=True, limit=1)

    # visualize_cam(model, DATASET_INDEX, dataset_prefix='swedish_leaf', class_id=0)
