import keras
from keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gc
import time
from sklearn import metrics
from libs.Dataset import Dataset
# from scipy.ndimage import imread
from imageio import imread
from skimage.transform import resize


class Network:
    def __init__(self, n_downsample_interval=2,
                 is_res=False,
                 is_pool=False,
                 n_layers=3,
                 n_layer_sizes=[256,256,256],
                 n_kernel_sizes=[8,8,8],
                 n_blocks=0,
                 n_block_sizes=None,  # list
                 activation_name="relu"
                 ):

        self.n_downsample_interval = n_downsample_interval
        self.is_res = is_res
        self.is_pool = is_pool
        self.n_layers = n_layers
        self.n_layer_sizes = n_layer_sizes
        self.n_kernel_sizes = n_kernel_sizes
        self.n_blocks = n_blocks
        self.n_block_sizes = n_block_sizes
        self.activation_name = activation_name

        if len(n_layer_sizes) != n_layers or len(n_kernel_sizes) != n_layers:
            ValueError("n_layer_sizes and n_kernel_sizes must be lists with length equaling to n_layers!")

        if (is_res is True) and (n_blocks == 0 or n_block_sizes is None):
            ValueError("If is_res is True, n_blocks and n_block_size must > 0!")

        if is_res is True and len(n_block_sizes) != n_blocks:
            ValueError("If is_res is True, n_block_sizes must be a list with length equaling to n_blocks!")

        if is_res is True and sum(n_block_sizes) != n_layers:
            ValueError("If is_res is True, n_blocks times n_block_size must equal to n_layers!")

        return

    @staticmethod
    def down_samples(input_signal, interval=2):
        n_raw_samples = input_signal.shape[1]  # The length of each segment
        n_new_samples = int(np.floor(n_raw_samples / interval))
        down_sampled_loc = np.linspace(0, (n_new_samples - 1) * interval, n_new_samples, dtype=np.int)
        down_sampled_signal = input_signal[:, down_sampled_loc, :]
        return down_sampled_signal, n_new_samples

    def train(self, train_dataset,
              valid_dataset = None,
              epochs = 100,
              lr = 0.00001,
              decay=0.0001,
              is_plot = False,
              is_save = False,
              save_path = "../KerasModels/",
              save_name = "model.h5"):

        n_samples = train_dataset.signal_samples.shape[0]
        n_features = train_dataset.signal_samples.shape[1]

        train_samples = train_dataset.signal_samples.reshape((n_samples, n_features, 1))
        train_samples, n_features = Network.down_samples(train_samples, self.n_downsample_interval)
        train_labels = keras.utils.to_categorical(train_dataset.labels, 2)
        if valid_dataset is not None:
            n_valid_samples = valid_dataset.signal_samples.shape[0]
            n_valid_features = valid_dataset.signal_samples.shape[1]
            valid_samples = valid_dataset.signal_samples.reshape((n_valid_samples, n_valid_features, 1))
            valid_samples, n_valid_features = Network.down_samples(valid_samples, self.n_downsample_interval)
            valid_labels = keras.utils.to_categorical(valid_dataset.labels, 2)

        if self.is_res is False and self.is_pool is False:
            model = keras.models.Sequential()
            for i_layer in range(self.n_layers):
                if i_layer == 0:
                    model.add(layers.Conv1D(self.n_layer_sizes[i_layer], kernel_size=self.n_kernel_sizes[i_layer],
                                            strides=1, padding="same", input_shape=(n_features, 1)))
                else:
                    model.add(layers.Conv1D(self.n_layer_sizes[i_layer], kernel_size=self.n_kernel_sizes[i_layer],
                                            strides=1, padding="same"))
                model.add(layers.Activation(self.activation_name))
            model.add(layers.Flatten())
            model.add(layers.Dense(512))
            model.add(layers.Activation(self.activation_name))
            model.add(layers.Dense(2))
            model.add(layers.Activation("softmax"))
        elif self.is_res is False and self.is_pool is True:
            model = keras.models.Sequential()
            for i_layer in range(self.n_layers):
                if i_layer == 0:
                    model.add(layers.Conv1D(self.n_layer_sizes[i_layer], kernel_size=self.n_kernel_sizes[i_layer],
                                            strides=1, padding="same", input_shape=(n_features, 1)))
                else:
                    model.add(layers.Conv1D(self.n_layer_sizes[i_layer], kernel_size=self.n_kernel_sizes[i_layer],
                                            strides=1, padding="same"))
                model.add(layers.MaxPool1D(pool_size=4, strides=4, padding="same"))
                model.add(layers.Activation(self.activation_name))
            model.add(layers.Flatten())
            model.add(layers.Dense(512))
            model.add(layers.Activation(self.activation_name))
            model.add(layers.Dense(2))
            model.add(layers.Activation("softmax"))
        elif self.is_res is True and self.is_pool is False:
            inputs = layers.Input(shape=(n_features, 1))

            def block(input_layer, n_block_size, n_block_layer_sizes, n_block_kernel_sizes, activation_name):
                input_reshaped_layer = layers.Conv1D(n_block_layer_sizes[-1], kernel_size=1,
                                                     strides=1, padding="same")(input_layer)
                conv_layer = layers.Conv1D(n_block_layer_sizes[0], kernel_size=n_block_kernel_sizes[0],
                                           strides=1, padding="same")(input_layer)
                for i_layer in range(1, n_block_size):
                    conv_layer = layers.Conv1D(n_block_layer_sizes[i_layer], kernel_size=n_block_kernel_sizes[i_layer],
                                               strides=1, padding="same")(conv_layer)
                add_layer = layers.add([input_reshaped_layer, conv_layer])
                output_layer = layers.Activation(activation=activation_name)(add_layer)
                return output_layer

            for i_block in range(self.n_blocks):
                if i_block == 0:
                    n_block_layer_sizes = self.n_layer_sizes[0:self.n_block_sizes[0]]
                    n_block_kernel_sizes = self.n_kernel_sizes[0:self.n_block_sizes[0]]
                    resnet = block(inputs, self.n_block_sizes[i_block],
                                   n_block_layer_sizes, n_block_kernel_sizes, self.activation_name)
                else:
                    n_block_layer_sizes = self.n_layer_sizes[sum(self.n_block_sizes[0:i_block]):sum(self.n_block_sizes[0:i_block+1])]
                    n_block_kernel_sizes = self.n_kernel_sizes[sum(self.n_block_sizes[0:i_block]):sum(self.n_block_sizes[0:i_block+1])]
                    resnet = block(resnet, self.n_block_sizes[i_block],
                                   n_block_layer_sizes, n_block_kernel_sizes, self.activation_name)

            pool_layer = layers.MaxPool1D(pool_size=4, strides=4, padding="same")(resnet)
            flat_layer = layers.Flatten()(pool_layer)
            fc_layer_1 = layers.Dense(512, activation=self.activation_name)(flat_layer)
            fc_layer_2 = layers.Dense(2, activation="softmax")(fc_layer_1)

            model = keras.Model(inputs=inputs, outputs=fc_layer_2)
        elif self.is_res is True and self.is_pool is True:
            inputs = layers.Input(shape=(n_features, 1))

            def block(input_layer, n_block_size, n_block_layer_sizes, n_block_kernel_sizes, activation_name):
                input_reshaped_layer = layers.Conv1D(n_block_layer_sizes[-1], kernel_size=1,
                                                     strides=1, padding="same")(input_layer)
                input_reshaped_layer = layers.MaxPool1D(pool_size=4, strides=4, padding="same")(input_reshaped_layer)
                conv_layer = layers.Conv1D(n_block_layer_sizes[0], kernel_size=n_block_kernel_sizes[0],
                                           strides=1, padding="same")(input_layer)
                conv_layer = layers.MaxPool1D(pool_size=4, strides=4, padding="same")(conv_layer)
                for i_layer in range(1, n_block_size):
                    conv_layer = layers.Conv1D(n_block_layer_sizes[i_layer], kernel_size=n_block_kernel_sizes[i_layer],
                                               strides=1, padding="same")(conv_layer)
                    conv_layer = layers.MaxPool1D(pool_size=4, strides=4, padding="same")(conv_layer)
                add_layer = layers.add([input_reshaped_layer, conv_layer])
                output_layer = layers.Activation(activation=activation_name)(add_layer)
                return output_layer

            for i_block in range(self.n_blocks):
                if i_block == 0:
                    n_block_layer_sizes = self.n_layer_sizes[0:self.n_block_sizes[0]]
                    n_block_kernel_sizes = self.n_kernel_sizes[0:self.n_block_sizes[0]]
                    resnet = block(inputs, self.n_block_sizes[i_block],
                                   n_block_layer_sizes, n_block_kernel_sizes, self.activation_name)
                else:
                    n_block_layer_sizes = self.n_layer_sizes[sum(self.n_block_sizes[0:i_block]):sum(self.n_block_sizes[0:i_block+1])]
                    n_block_kernel_sizes = self.n_kernel_sizes[sum(self.n_block_sizes[0:i_block]):sum(self.n_block_sizes[0:i_block+1])]
                    resnet = block(resnet, self.n_block_sizes[i_block],
                                   n_block_layer_sizes, n_block_kernel_sizes, self.activation_name)

            pool_layer = layers.MaxPool1D(pool_size=4, strides=4, padding="same")(resnet)
            flat_layer = layers.Flatten()(pool_layer)
            fc_layer_1 = layers.Dense(512, activation=self.activation_name)(flat_layer)
            fc_layer_2 = layers.Dense(2, activation="softmax")(fc_layer_1)

            model = keras.Model(inputs=inputs, outputs=fc_layer_2)

        model.summary()

        adam_op = keras.optimizers.Adam(lr=lr, decay=decay)
        # # adam_op = keras.optimizers.Adam(lr=0.0000001, decay=0)

        model.compile(optimizer=adam_op, loss="categorical_crossentropy", metrics=["acc"])

        if is_save:
            if os.path.exists(save_path):
                pass
            else:
                os.makedirs(save_path)
            model_filename = save_path + save_name
            check_point = keras.callbacks.ModelCheckpoint(model_filename, monitor="val_loss", verbose=1,
                                                          save_best_only=True, mode="min")
            callback_list = [check_point]
            if valid_dataset is not None:
                hist = model.fit(train_samples, train_labels, batch_size=16, epochs=epochs, verbose=2,
                                 validation_data=(valid_samples, valid_labels),
                                 shuffle=True, callbacks=callback_list)
            else:
                hist = model.fit(train_samples, train_labels, batch_size=16, epochs=epochs, verbose=2,
                                 validation_split=0.3,
                                 shuffle=True, callbacks=callback_list)
        else:
            if valid_dataset is not None:
                hist = model.fit(train_samples, train_labels, batch_size=16, epochs=epochs, verbose=2,
                                 validation_split=(valid_samples, valid_labels),
                                 shuffle=True)
            else:
                hist = model.fit(train_samples, train_labels, batch_size=16, epochs=epochs, verbose=2,
                                 validation_split=0.3,
                                 shuffle=True)

        loss_train = hist.history["loss"]
        acc_train = hist.history["acc"]
        loss_valid = hist.history["val_loss"]
        acc_valid = hist.history["val_acc"]

        if is_save:
            np.savetxt(save_path + "loss_train.txt", np.array(loss_train))
            np.savetxt(save_path + "loss_valid.txt", np.array(loss_valid))
            np.savetxt(save_path + "acc_train.txt", np.array(acc_train))
            np.savetxt(save_path + "acc_valid.txt", np.array(acc_valid))
        if is_plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(loss_train)
            plt.plot(loss_valid)
            plt.subplot(1, 2, 2)
            plt.plot(acc_train)
            plt.plot(acc_valid)
            plt.show()
        return loss_train, loss_valid, acc_train, acc_valid

    def evaluate(self, test_dataset, model_name):
        n_samples = test_dataset.signal_samples.shape[0]
        n_features = test_dataset.signal_samples.shape[1]
        test_samples = test_dataset.signal_samples.reshape((n_samples, n_features, 1))
        test_samples, n_features = Network.down_samples(test_samples, interval=self.n_downsample_interval)
        test_labels = keras.utils.to_categorical(test_dataset.labels, 2)
        # model_path = "/home/zhaogy/GitHubWorkspace/mad-py/KerasModels/"
        model_path = "../KerasModels/"
        model_file = model_path + model_name + "/model.h5"
        model = keras.models.load_model(model_file)
        loss, acc = model.evaluate(test_samples, test_labels, verbose=0)
        pred = model.predict(test_samples, verbose=0)

        n_targets = np.sum(test_dataset.labels)
        n_alarms = np.sum(np.argmax(pred, axis=1))

        n_correctly_detected_targets = 0
        n_correctly_non_alarms = 0
        n_missing_targets = 0
        n_false_alarms = 0
        tc_total = 0
        for sample_idx in range(n_samples):
            a_sample = test_samples[sample_idx]
            st_time = time.time()
            model.predict(a_sample[np.newaxis, :], verbose=0)
            ed_time = time.time()
            tc_total = tc_total + ed_time - st_time
            if test_dataset.labels[sample_idx] == 1:
                if np.argmax(pred[sample_idx]) == 1:
                    n_correctly_detected_targets += 1
                else:
                    n_missing_targets += 1
            else:
                if np.argmax(pred[sample_idx]) == 1:
                    n_false_alarms += 1
                else:
                    n_correctly_non_alarms += 1
        missing_rate = n_missing_targets / n_targets
        detection_rate = n_correctly_detected_targets / n_targets
        false_rate = n_false_alarms / n_alarms

        fpr, tpr, thrs = metrics.roc_curve(test_labels.ravel(), pred.ravel())
        auc = metrics.auc(fpr, tpr)

        time_consumption = 1000 * tc_total / n_samples
        criteria = {'Accuracy': acc, 'MissingRate': missing_rate, 'DetectionRate': detection_rate,
                    'FalseAlarmRate': false_rate, 'AUC': auc, 'TimeConsuming': time_consumption,
                    'FPR': fpr, 'TPR': tpr}

        return pred, criteria



