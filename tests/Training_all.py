import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from libs.ConvNetworks import Network
from Dataset import Dataset


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)

    train_filename = "20210406-17"
    test_filename = "20210406-18"

    train_dataset = Dataset()
    train_dataset.load(train_filename)
    test_dataset = Dataset()
    test_dataset.load(test_filename)

    epochs = 3000

    ###ResNet
    width = 128
    block_size = 3
    n_blocks = 3
    n_layers = n_blocks*block_size
    lr_decay_range = [[0.000001, 0.00001], [0.00001, 0.0001]]
    for lr_decay in lr_decay_range:
        lr = lr_decay[0]
        decay = lr_decay[1]
        nn = Network(n_downsample_interval=5, is_res=True, is_pool=False, activation_name="relu",
                     n_blocks=n_blocks, n_block_sizes=[block_size] * n_blocks,
                     n_layers=n_layers, n_layer_sizes=[width] * n_layers, n_kernel_sizes=[8] * n_layers)
        save_path = "../KerasModels/" + train_filename + "-ResNet-" + str(width) + "-" + str(block_size) + "-" + str(
            n_blocks) + "--" + str(lr) + "-" + str(decay) + "/"
        result = nn.train(train_dataset, valid_dataset=None, epochs=epochs, lr=lr, decay=decay, is_plot=False, is_save=True,
                          save_path=save_path)
        del nn, save_path, result
        gc.collect()

    width = 64
    block_size = 3
    n_blocks = 3
    n_layers = n_blocks * block_size
    lr_decay_range = [[0.000001, 0.00001], [0.00001, 0.0001]]
    for lr_decay in lr_decay_range:
        lr = lr_decay[0]
        decay = lr_decay[1]
        nn = Network(n_downsample_interval=5, is_res=True, is_pool=False, activation_name="relu",
                     n_blocks=n_blocks, n_block_sizes=[block_size] * n_blocks,
                     n_layers=n_layers, n_layer_sizes=[width] * n_layers, n_kernel_sizes=[8] * n_layers)
        save_path = "../KerasModels/" + train_filename + "-ResNet-" + str(width) + "-" + str(block_size) + "-" + str(
            n_blocks) + "--" + str(lr) + "-" + str(decay) + "/"
        result = nn.train(train_dataset, valid_dataset=None, epochs=epochs, lr=lr, decay=decay, is_plot=False,
                          is_save=True,
                          save_path=save_path)
        del nn, save_path, result
        gc.collect()

    width = 64
    block_size = 3
    n_blocks = 2
    n_layers = n_blocks * block_size
    lr_decay_range = [[0.000001, 0.00001], [0.00001, 0.0001]]
    for lr_decay in lr_decay_range:
        lr = lr_decay[0]
        decay = lr_decay[1]
        nn = Network(n_downsample_interval=5, is_res=True, is_pool=False, activation_name="relu",
                     n_blocks=n_blocks, n_block_sizes=[block_size] * n_blocks,
                     n_layers=n_layers, n_layer_sizes=[width] * n_layers, n_kernel_sizes=[8] * n_layers)
        save_path = "../KerasModels/" + train_filename + "-ResNet-" + str(width) + "-" + str(block_size) + "-" + str(
            n_blocks) + "--" + str(lr) + "-" + str(decay) + "/"
        result = nn.train(train_dataset, valid_dataset=None, epochs=epochs, lr=lr, decay=decay, is_plot=False,
                          is_save=True,
                          save_path=save_path)
        del nn, save_path, result
        gc.collect()

    width = 64
    block_size = 3
    n_blocks = 4
    n_layers = n_blocks * block_size
    lr_decay_range = [[0.000001, 0.00001], [0.00001, 0.0001]]
    for lr_decay in lr_decay_range:
        lr = lr_decay[0]
        decay = lr_decay[1]
        nn = Network(n_downsample_interval=5, is_res=True, is_pool=False, activation_name="relu",
                     n_blocks=n_blocks, n_block_sizes=[block_size] * n_blocks,
                     n_layers=n_layers, n_layer_sizes=[width] * n_layers, n_kernel_sizes=[8] * n_layers)
        save_path = "../KerasModels/" + train_filename + "-ResNet-" + str(width) + "-" + str(block_size) + "-" + str(
            n_blocks) + "--" + str(lr) + "-" + str(decay) + "/"
        result = nn.train(train_dataset, valid_dataset=None, epochs=epochs, lr=lr, decay=decay, is_plot=False,
                          is_save=True,
                          save_path=save_path)
        del nn, save_path, result
        gc.collect()


if __name__ == "__main__":
    main()
