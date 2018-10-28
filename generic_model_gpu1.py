from utils.constants import max_seq_len, nb_classes
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualize_context_vector, visualize_cam
from utils.model_utils import lstm_fcn_model, alstm_fcn_model
from utils.model_utils import cnn_raw_model, cnn_dtwfeatures_model, cnn_earlyfusion_model, cnn_midfusion_model, cnn_latefusion_model

import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    dataset = sys.argv[1]
    method = sys.argv[2]
    proto_num = int(sys.argv[3])
    nb_cnn = int(sys.argv[4])

    max_seq_lenth = max_seq_len(dataset)
    nb_class = nb_classes(dataset)

    #model = lstm_fcn_model(proto_num, max_seq_lenth, nb_class)
    #model = alstm_fcn_model(proto_num, max_seq_lenth, nb_class)

    #model = cnn_raw_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_dtwfeatures_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_earlyfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    model = cnn_midfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = cnn_latefusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)


    train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=100000, batch_size=50, learning_rate=0.0001, early_stop=False, balance_classes=False)
    #train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=28000, batch_size=64, learning_rate=0.001, early_stop=True)

    evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="loss")

    evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="val_acc")

