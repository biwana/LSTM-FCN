import utils.dtw as dtw
import time
import numpy as np
import os
import csv
import sys


if __name__ == "__main__":
    dataset = sys.argv[1]
    length = int(sys.argv[2]) # 50
    depth = int(sys.argv[3]) # 2

    print("Starting: {}".format(dataset))
    # load settings

    full_data_file = os.path.join("data", dataset + "-data.txt")
    # load data
    train_data = np.genfromtxt(full_data_file, delimiter=' ').reshape((-1, length, depth))


    train_number = np.shape(train_data)[0]

    fileloc = os.path.join("data", "all-"+dataset + "-dtw-matrix.txt")

    lap = time.time()

    with open(fileloc, 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE, delimiter=" ")
        for t1 in range(train_number):
            writeline = np.zeros((train_number))
            for t2 in range(train_number):
                writeline[t2] = dtw.dtw(train_data[t1], train_data[t2], extended=False)
            writer.writerow(writeline)
            if t1 % (train_number // 20) == 0:
                print(str(t1))
                #print("step: %s time: %s" % (str(t1), str(round(time.time()-lap),1)))
                lap = time.time()

print("Done")
