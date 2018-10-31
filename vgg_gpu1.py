from utils.constants import max_seq_len, nb_classes
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualize_context_vector, visualize_cam
from utils.model_utils import lstm_fcn_model, alstm_fcn_model
from utils.model_utils import cnn_raw_model, cnn_dtwfeatures_model, cnn_earlyfusion_model, cnn_midfusion_model, cnn_latefusion_model
from utils.model_utils import vgg_raw_model, vgg_dtwfeatures_model, vgg_earlyfusion_model, vgg_midfusion_model, vgg_latefusion_model

import sys
import math
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    model = vgg_midfusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)
    #model = vgg_latefusion_model(nb_cnn, proto_num, max_seq_lenth, nb_class)


    train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=50000, batch_size=50, learning_rate=0.0001, early_stop=False, balance_classes=False, run_ver='vgg_')
    #train_model(model, dataset, method, proto_num, dataset_prefix=dataset, nb_iterations=28000, batch_size=64, learning_rate=0.001, early_stop=True)

    acc = evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="vgg_loss")
    np.savetxt("output/vgg/vgg-%s-%s-%s-loss-%s" % (dataset, method, str(proto_num), str(acc)), [acc])

    acc = evaluate_model(model, dataset, method, proto_num, dataset_prefix=dataset, batch_size=50, checkpoint_prefix="vgg_val_acc")
    np.savetxt("output/vgg/vgg-%s-%s-%s-vacc-%s" % (dataset, method, str(proto_num), str(acc)), [acc])
