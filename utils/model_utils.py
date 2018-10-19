from keras.models import Model
from keras.layers import Input, Dense, multiply, concatenate, Activation, Lambda
from keras.layers import PReLU, LSTM
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import MaxPooling1D, Flatten

from utils.layer_utils import AttentionLSTM

TRAINABLE = True

def slice_seq(x):
    return x[:, :1]

def slice_dtw(x):
    return x[:, 1:]

def lstm_fcn_model(proto_num, max_seq_lenth, nb_class):
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
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


def alstm_fcn_model(proto_num, max_seq_lenth, nb_class):
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

def cnn_raw_model(nb_cnn, proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x = Permute((2, 1))(ip1)

    for i in range(nb_cnn):
        x = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

def cnn_dtwfeatures_model(nb_cnn, proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x = Permute((2, 1))(ip2)

    for i in range(nb_cnn):
        x = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

def cnn_earlyfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    x = Permute((2, 1))(ip)

    for i in range(nb_cnn):
        x = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

def cnn_midfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x1 = Permute((2, 1))(ip1)
    x2 = Permute((2, 1))(ip2)

    for i in range(nb_cnn):
        x1 = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)

        x2 = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling1D(pool_size=2)(x2)


    x = concatenate([x1, x2])

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


def cnn_latefusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class):
    ip = Input(shape=(1+proto_num, max_seq_lenth))

    ip1 = Lambda(slice_seq)(ip)
    ip2 = Lambda(slice_dtw)(ip)

    x1 = Permute((2, 1))(ip1)
    x2 = Permute((2, 1))(ip2)

    for i in range(nb_cnn):
        x1 = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)

        x2 = Conv1D(256, 3, padding='same', kernel_initializer='he_uniform')(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling1D(pool_size=2)(x2)

    x1 = Flatten()(x1)
    x2 = Flatten()(x2)

    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    x2 = Dense(1024, activation='relu')(x1)
    x2 = Dropout(0.5)(x1)

    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)

    x = concatenate([x1, x2])

    out = Dense(nb_class, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model
