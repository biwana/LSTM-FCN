import utils.dtw as dtw
import time
import numpy as np
import math
import csv
import os
import sys
from utils.constants import nb_classes, class_modifier_add, class_modifier_multi, max_seq_len
from utils.proto_select import selector_selector, random_selection, center_selection, k_centers_selection, border_selection, spanning_selection


def get_dtwfeatures(proto_data, proto_number, local_sample):
    local_sample_length = np.shape(local_sample)[0]
    features = np.zeros((local_sample_length, proto_number))
    for prototype in range(proto_number):
        local_proto = proto_data[prototype]
        output, cost, DTW, path = dtw.dtw(local_proto, local_sample, extended=True)

        for f in range(local_sample_length):
            features[f, prototype] = cost[path[0][f]][path[1][f]]
    return features

def get_one_fold(number, test_ratio, fold):
    indices = np.arange(number)
    test_start = fold * int(test_ratio * float(number))
    test_end = (fold+1) * int(test_ratio * float(number))
    testset = indices[test_start:test_end]
    testset[::-1].sort()
    for pl in testset:
        indices = np.delete(indices, pl, 0)
    trainset = indices
    return trainset, testset

def read_dtw_matrix(version, test_ratio, fold):
    if not os.path.exists(os.path.join("data", "all-"+version+"-dtw-matrix.txt")):
        exit("Please run cross_dtw_full.py first")
    full = np.genfromtxt(os.path.join("data", "all-"+version+"-dtw-matrix.txt"), delimiter=' ')
    number = full.shape[0]
    trainset, testset = get_one_fold(number, test_ratio, fold)
    ret = full[trainset]
    return ret[:,trainset]



if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Error, Syntax: {0} [version] [prototype selection] [classwise/independent] [prototype number] [num classes] [dimensions] [fold]".format(sys.argv[0]))
        exit()
    version = sys.argv[1]
    selection = sys.argv[2]
    classwise = sys.argv[3]
    proto_number = int(sys.argv[4])
    no_classes = int(sys.argv[5])
    dimensions = int(sys.argv[6])
    fold = int(sys.argv[7])
    test_ratio = 0.1

    print("Starting: {}".format(version))

    # load settings
    full_train_file = os.path.join("data", version + "-data.txt")
    full_train_label_file = os.path.join("data", version + "-labels.txt")
    # load data
    full_train = np.genfromtxt(full_train_file, delimiter=' ')
    full_train_label = np.genfromtxt(full_train_label_file, delimiter=' ')

    train_max = np.max(full_train)
    print(train_max)
    train_min = np.min(full_train)
    print(train_min)

    train_data = 2. * (full_train - train_min) / (train_max - train_min) - 1.
    train_labels = full_train_label[:,1]

    seq_length = int(np.shape(train_data)[1] / dimensions)
    total_number = np.shape(train_labels)[0]


    trainset, testset = get_one_fold(total_number, test_ratio, fold)
    test_data = train_data[testset]
    test_labels = train_labels[testset]
    train_data = train_data[trainset]
    train_labels = train_labels[trainset]

    train_number = np.shape(train_labels)[0]
    test_number = np.shape(test_labels)[0]

    train_data = train_data.reshape((-1,seq_length, dimensions))
    test_data = test_data.reshape((-1,seq_length, dimensions))

    distances = train_number if selection == "random" else read_dtw_matrix(version, test_ratio, fold)

    print(np.shape(train_labels))
    print(np.shape(distances))

    if classwise == "classwise":
        proto_loc = np.zeros(0, dtype=np.int32)
        proto_factor = int(proto_number / no_classes)
        for c in range(no_classes):
            cw = np.where(train_labels == c)[0]
            if selection == "random":
                cw_distances = []
            else:
                cw_distances = distances[cw]
                cw_distances = cw_distances[:,cw]
            cw_proto = selector_selector(selection, proto_factor, cw_distances)
            proto_loc = np.append(proto_loc, cw[cw_proto])
    else:
        proto_loc = selector_selector(selection, proto_number, distances)

    proto_data = train_data[proto_loc]
    print(proto_loc)
    #exit()
    #print("Selection Done.")

    # sorts the prototypes so deletion happens in reverse order and doesn't interfere with indices
    #proto_loc[::-1].sort()

    # remove prototypes from training data
    #for pl in proto_loc:
    #    train_data = np.delete(train_data, pl, 0)
    #    train_labels = np.delete(train_labels, pl, 0)

    # start generation
    test_label_fileloc = os.path.join("data", "fold{}-test-label-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    test_raw_fileloc = os.path.join("data", "fold{}-raw-test-data-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    test_dtw_fileloc = os.path.join("data", "fold{}-dtw_features-test-data-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    test_combined_fileloc = os.path.join("data", "fold{}-dtw_features-plus-raw-test-data-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    train_label_fileloc = os.path.join("data", "fold{}-train-label-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    train_raw_fileloc = os.path.join("data", "fold{}-raw-train-data-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    train_dtw_fileloc = os.path.join("data", "fold{}-dtw_features-train-data-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))
    train_combined_fileloc = os.path.join("data", "fold{}-dtw_features-plus-raw-train-data-{}-{}-{}-{}.txt".format(fold, version, selection, classwise, proto_number))

    # test set
    with open(test_label_fileloc, 'w') as test_label_file, open(test_raw_fileloc, 'w') as test_raw_file, open(
            test_dtw_fileloc, 'w') as test_dtw_file, open(test_combined_fileloc, 'w') as test_combined_file:
        writer_test_label = csv.writer(test_label_file, quoting=csv.QUOTE_NONE, delimiter=" ")
        writer_test_raw = csv.writer(test_raw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
        writer_test_dtw = csv.writer(test_dtw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
        writer_test_combined = csv.writer(test_combined_file, quoting=csv.QUOTE_NONE, delimiter=" ")

        for sample in range(test_number):
            local_sample = test_data[sample]
            features = get_dtwfeatures(proto_data, proto_number, local_sample)

            class_value = test_labels[sample]

            # write files
            feature_flat = features.reshape(seq_length * proto_number)
            local_sample_flat = local_sample.reshape(seq_length * dimensions)
            writer_test_raw.writerow(local_sample_flat)
            writer_test_dtw.writerow(feature_flat)
            writer_test_combined.writerow(np.append(local_sample_flat, feature_flat))
            writer_test_label.writerow(["{}-{}_test.png".format(class_value, sample), class_value])
            if sample % (test_number // 16) == 0:
                print("{} {}%: Test < {} Done".format(version, str(round(100. * sample / test_number, 1)),str(sample)))
    print("{}: Test Done".format(version))

    # train set
    with open(train_label_fileloc, 'w') as train_label_file, open(train_raw_fileloc, 'w') as train_raw_file, open(
            train_dtw_fileloc, 'w') as train_dtw_file, open(train_combined_fileloc, 'w') as train_combined_file:
        writer_train_label = csv.writer(train_label_file, quoting=csv.QUOTE_NONE, delimiter=" ")
        writer_train_raw = csv.writer(train_raw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
        writer_train_dtw = csv.writer(train_dtw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
        writer_train_combined = csv.writer(train_combined_file, quoting=csv.QUOTE_NONE, delimiter=" ")

        for sample in range(train_number):
            local_sample = train_data[sample]
            features = get_dtwfeatures(proto_data, proto_number, local_sample)

            class_value = train_labels[sample]

            # write files
            feature_flat = features.reshape(seq_length * proto_number)
            local_sample_flat = local_sample.reshape(seq_length * dimensions)
            writer_train_raw.writerow(local_sample_flat)
            writer_train_dtw.writerow(feature_flat)
            writer_train_combined.writerow(np.append(local_sample_flat, feature_flat))
            writer_train_label.writerow(["{}-{}_train.png".format(class_value, sample), class_value])
            
            if sample % (train_number // 16) == 0:
                print("{} {}%: Training < {} Done".format(version, str(round(100. * sample / train_number,1)),str(sample)))
    print("{}: Training Done".format(version))


print("Done")
