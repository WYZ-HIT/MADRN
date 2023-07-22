import warnings
import gc
from  libs.Dataset import Dataset
from libs.ConvNetworks import Network as CN
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import copy
import keras
import os
import tensorflow as tf

warnings.filterwarnings("ignore")


# ===================================================================
# data for table testing the width and depth
gc.collect()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
# tf.random.set_seed(seed)

train_filename = "20210409-17"
test_filename = "20210409-18"
train_dataset = Dataset()
train_dataset.load(train_filename)
test_dataset = Dataset()
test_dataset.load(test_filename)
net = CN(n_downsample_interval=5)

##model
lr = 0.000001
decay = 0.00001
# lr = 0.00001
# decay = 0.0001

test_result_net_64_3_3 = net.evaluate(test_dataset, train_filename + "-ResNet-64-3-3" + "--" + str(lr) + "-" + str(decay))
test_result_net_64_3_2 = net.evaluate(test_dataset, train_filename + "-ResNet-64-3-2" + "--" + str(lr) + "-" + str(decay))
test_result_net_64_3_4 = net.evaluate(test_dataset, train_filename + "-ResNet-64-3-4" + "--" + str(lr) + "-" + str(decay))
test_result_net_128_3_3 = net.evaluate(test_dataset, train_filename + "-ResNet-128-3-3" + "--" + str(lr) + "-" + str(decay))
train_result_net_64_3_3 = net.evaluate(train_dataset, train_filename + "-ResNet-64-3-3" + "--" + str(lr) + "-" + str(decay))
train_result_net_64_3_2 = net.evaluate(train_dataset, train_filename + "-ResNet-64-3-2" + "--" + str(lr) + "-" + str(decay))
train_result_net_64_3_4 = net.evaluate(train_dataset, train_filename + "-ResNet-64-3-4" + "--" + str(lr) + "-" + str(decay))
train_result_net_128_3_3 = net.evaluate(train_dataset, train_filename + "-ResNet-128-3-3" + "--" + str(lr) + "-" + str(decay))
train_auc_net_64_3_3 = round(train_result_net_64_3_3[1]['AUC'], 4)
train_tc_net_64_3_3 = round(train_result_net_64_3_3[1]['TimeConsuming'], 4)
train_auc_net_64_3_2 = round(train_result_net_64_3_2[1]['AUC'], 4)
train_tc_net_64_3_2 = round(train_result_net_64_3_2[1]['TimeConsuming'], 4)
train_auc_net_64_3_4 = round(train_result_net_64_3_4[1]['AUC'], 4)
train_tc_net_64_3_4 = round(train_result_net_64_3_4[1]['TimeConsuming'], 4)
train_auc_net_128_3_3 = round(train_result_net_128_3_3[1]['AUC'], 4)
train_tc_net_128_3_3 = round(train_result_net_128_3_3[1]['TimeConsuming'], 4)
test_auc_net_64_3_3 = round(test_result_net_64_3_3[1]['AUC'], 4)
test_tc_net_64_3_3 = round(test_result_net_64_3_3[1]['TimeConsuming'], 4)
test_auc_net_64_3_2 = round(test_result_net_64_3_2[1]['AUC'], 4)
test_tc_net_64_3_2 = round(test_result_net_64_3_2[1]['TimeConsuming'], 4)
test_auc_net_64_3_4 = round(test_result_net_64_3_4[1]['AUC'], 4)
test_tc_net_64_3_4 = round(test_result_net_64_3_4[1]['TimeConsuming'], 4)
test_auc_net_128_3_3 = round(test_result_net_128_3_3[1]['AUC'], 4)
test_tc_net_128_3_3 = round(test_result_net_128_3_3[1]['TimeConsuming'], 4)
print("Net: 64-3-3; AUC-train: " + str(train_auc_net_64_3_3) + "; Time-train: " + str(train_tc_net_64_3_3) +
      "; AUC-test: " + str(test_auc_net_64_3_3) + "; Time-test: " + str(test_tc_net_64_3_3))
print("Net: 64-3-2; AUC-train: " + str(train_auc_net_64_3_2) + "; Time-train: " + str(train_tc_net_64_3_2) +
      "; AUC-test: " + str(test_auc_net_64_3_2) + "; Time-test: " + str(test_tc_net_64_3_2))
print("Net: 64-3-4; AUC-train: " + str(train_auc_net_64_3_4) + "; Time-train: " + str(train_tc_net_64_3_4) +
      "; AUC-test: " + str(test_auc_net_64_3_4) + "; Time-test: " + str(test_tc_net_64_3_4))
print("Net: 128-3-3; AUC-train: " + str(train_auc_net_128_3_3) + "; Time-train: " + str(train_tc_net_128_3_3) +
      "; AUC-test: " + str(test_auc_net_128_3_3) + "; Time-test: " + str(test_tc_net_128_3_3))

saveDict = {'net-64-3-3_auc_train': train_auc_net_64_3_3, 'net-64-3-3_time_train': train_tc_net_64_3_3,
            'net-64-3-3_auc_test': test_auc_net_64_3_3, 'net-64-3-3_time_test': test_tc_net_64_3_3,
            'net-64-3-2_auc_train': train_auc_net_64_3_2, 'net-64-3-2-time_train': train_tc_net_64_3_2,
            'net-64-3-2_auc_test': test_auc_net_64_3_2, 'net-64-3-2_time_test': test_tc_net_64_3_2,
            'net-64-3-4_auc_train': train_auc_net_64_3_4, 'net-64-3-4_time_train': train_tc_net_64_3_4,
            'net-64-3-4_auc_test': test_auc_net_64_3_4, 'net-64-3-4_time_test': test_tc_net_64_3_4,
            'net-128-3-3_auc_train': train_auc_net_128_3_3, 'net-128-3-3_time_train': train_tc_net_128_3_3,
            'net-128-3-3_auc_test': test_auc_net_128_3_3, 'net-128-3-3_time_test': test_tc_net_128_3_3
            }

savemat_name = "net_width_depth_gpu" + "-"+train_filename+"-" + str(lr) + "-" + str(decay) + ".mat"
sio.savemat(savemat_name, saveDict)

