# what we want
#   Output: softmax, 1 hot encoding (categorical_crossentropy)
#   Early stopping technique - 20% of training for validation
#   Data: training - optdigits.tra (3823), testing - optdigits.tes (1797)
#
# QUESTION 1
#   Fully-connected feed-forward NN
#   Experiment: different hyperparameters
#   (#hidden layers/size, learning, momentum, input scaling aka normal/standardization, etc)
#   Compare: classification accuracy, convergence speed
#   Report: model, hyperparams, classification & class accuracy, confusion matrix
#
#   Part a
#   Hidden layer: ReLU
#   Loss function: MSE VS cross-entropy
#
#   Part b
#   Loss function: cross-entropy
#   Hidden layer: tanh VS ReLU
#
# QUESTION 2
#   Convolutional networks CNN
#   Error function: cross-entropy
#   Hidden layer: ReLU
#   repeat experiment (might have different hyperparams: filter size)
#
#
#
# data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def normalize_data(method, x1, x2):
    """
    normalize data with chosen methods
    :param method:  max- divide by max-min
                    std- build in sklearn
    :param x1: trainx
    :param x2: testx
    :return: normalized x1, x2
    """
    if method == "max":
        x1 = (x1 - x1.min())/(x1.max()-x1.min())
        x2 = (x2 - x2.min()) / (x2.max() - x2.min())
    elif method == "std":
        x1 = StandardScaler().fit_transform(x1)
        x2 = StandardScaler().fit_transform(x2)
    return x1, x2


#   get hyperparam from user
def user_param(lr=0.01, mt=0.0, ep=50, hidl=1, hids=32, hidu=tf.nn.relu, lf='categorical_crossentropy'):
    """
    set hyperparameters
    :param lr: learning rate
    :param mt: momentum
    :param ep: epoch
    :param hidl: number of layer >1
    :param hids: hidden size
    :param hidu: hidden unit/activation function
    :param lf: loss function/error function
    :return: lr, mt, ep, hidl, hids, hidu, lf
    """

    while True:
        print('key in <0> for default parameter; or corresponding number to change the parameter (eg: 1=0.0001): ')
        param = input()
        print()
        if param != "0":
            if param[1] != "=":
                print('Invalid input\n')
            else:
                key = param[0]
                if key == "1":
                    lr = float(param[2:])
                elif key == "2":
                    mt = float(param[2:])
                elif key == "3":
                    ep = int(param[2:])
                elif key == "4":
                    hidl = int(param[2:])
                elif key == "5":
                    hids = int(param[2:])
                else:
                    print('Invalid input. Using default values.\n')
                return lr, mt, ep, hidl, hids, hidu, lf
        else:
            return lr, mt, ep, hidl, hids, hidu, lf


if __name__ == "__main__":
    # ------------------   EXPERIMENT VALUES ------------------ #
    #   set hyperparam
    lr_d = 0.01  # learning rate
    mt_d = 0.0  # momentum
    ep_d = 50  # epochs
    hidl_d = 1  # number of layers
    hids_d = 32  # hidden size
    hidu_d = tf.nn.relu  # hidden unit/activation function
    lf_d = 'categorical_crossentropy'  # loss function
    v = 0  # verbose
    debug = True

    # early stopping technique: stop training when no improvement in 3 consecutive epochs
    callback = tf.keras.callbacks.EarlyStopping(patience=3)
    #   load in data
    test_data = pd.read_csv('data/optdigits.tes', header=None)
    train_data = pd.read_csv('data/optdigits.tra', header=None)
    #   last attr is actual class
    test_X, test_Y = test_data.loc[:, 0:63], test_data.loc[:, 64]
    train_X, train_Y = train_data.loc[:, 0:63], train_data.loc[:, 64]
    test_total = len(test_data)
    train_total = len(train_data)

    #   to np array
    testy = np.array(test_Y)
    trainy = np.array(train_Y)

    #   convert into 1 hot encoding
    trainy = tf.keras.utils.to_categorical(trainy, 10)
    testy = tf.keras.utils.to_categorical(testy, 10)

    while True:
        #   Get user input
        print('key in <1> for feed forward network; <2> for CNN; <other key> to exit: ')
        question = input()
        print()

        #   FULLY CONNECTED FEED FORWARD NETWORK
        if question == "1":
            #   to np array
            testx = np.array(test_X)
            trainx = np.array(train_X)
            #   normalize data
            trainx, testx = normalize_data("max", trainx, testx)

            while True:
                print('key in <1> to test error function; <2> to test hidden units; <other key> other network: ')
                part = input()
                if part == "1" or part == "2":
                    print()
                    print('Default hyperparameter:\n<1>learning rate = ' + str(lr_d) + '\n<2>momentum = ' + str(mt_d))
                    print('<3>training epoch = ' + str(ep_d) + '\n<4>hidden layers = ' + str(hidl_d))
                    if part == "1":
                        print('<5>hidden size = ' + str(hids_d) + '\nhidden units = ReLU\n')
                    else:
                        print('<5>hidden size = ' + str(hids_d) + '\nerror function = cross entropy\n')
                    lr, mt, ep, hidl, hids, hidu, lf = user_param()

                ##
                #   MSE VS CROSS ENTROPY
                if part == "1":
                    #   make models
                    #   xent ----------------------------------------------------------------------
                    model1 = tf.keras.Sequential()
                    #   input layer
                    model1.add(keras.layers.Dense(64, input_shape=trainx[0].shape, activation=tf.nn.relu))
                    #   add layers
                    for i in range(hidl-1):
                        model1.add(keras.layers.Dense(hids, activation=tf.nn.relu))
                    #   output layer
                    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
                    model1.compile(loss='categorical_crossentropy',
                                   optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
                    history1 = model1.fit(trainx, trainy, epochs=ep, validation_split=0.2, verbose=v)

                    #   make model
                    #   MSE -----------------------------------------------------------------------
                    model2 = tf.keras.Sequential()
                    #   input layer
                    model2.add(keras.layers.Dense(64, input_shape=trainx[0].shape, activation=tf.nn.relu))
                    #   add layers
                    for i in range(hidl - 1):
                        keras.layers.Dense(hids, activation=tf.nn.relu)
                    #   output layer
                    model2.add(keras.layers.Dense(10, activation=tf.nn.softmax))
                    model2.compile(loss='mean_squared_error',
                                   optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
                    history2 = model2.fit(trainx, trainy, epochs=ep, validation_split=0.2, verbose=v)

                ##
                #   TANH VS RELU
                elif part == "2":
                    #   make model
                    #   tanh ----------------------------------------------------------------------
                    model1 = tf.keras.Sequential()
                    #   input layer
                    model1.add(keras.layers.Dense(64, input_shape=trainx[0].shape, activation=tf.nn.tanh))
                    #   add layers
                    for i in range(hidl - 1):
                        model1.add(keras.layers.Dense(hids, activation=tf.nn.tanh))
                    #   output layer
                    model1.add(keras.layers.Dense(10, activation=tf.nn.softmax))
                    model1.compile(loss='categorical_crossentropy',
                                   optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
                    history1 = model1.fit(trainx, trainy, epochs=ep, validation_split=0.2, verbose=v)

                    #   make model
                    #   relu ----------------------------------------------------------------------
                    model2 = tf.keras.Sequential()
                    #   input layer
                    model2.add(keras.layers.Dense(64, input_shape=trainx[0].shape, activation=tf.nn.relu))
                    #   add layers
                    for i in range(hidl):
                        model2.add(keras.layers.Dense(hids, activation=tf.nn.relu))
                    #   output layer
                    model2.add(keras.layers.Dense(10, activation=tf.nn.softmax))
                    model2.compile(loss='categorical_crossentropy',
                                   optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
                    history2 = model2.fit(trainx, trainy, epochs=ep, validation_split=0.2, verbose=v)

                else:
                    break

                #   output results
                # plt.plot(history1.epoch, history1.history['loss'], 'g')
                # plt.plot(history1.epoch, history1.history['val_loss'], 'r')
                print()
                print('Cross entropy (if testing on error function) / tanh (if testing on hidden units)')
                print('Result on train data: ')
                _, acc1_train = model1.evaluate(trainx, trainy)
                pred1_trainy = model1.predict(trainx)
                print(classification_report(trainy.argmax(axis=1), pred1_trainy.argmax(axis=1)))
                print('** See f1 score for accuracy and class accuracy\n')
                print('Confusion matrix: ')
                print(confusion_matrix(trainy.argmax(axis=1), pred1_trainy.argmax(axis=1)))
                print('Result on test data: ')
                _, acc1_test = model1.evaluate(testx, testy)
                pred1_testy = model1.predict(testx)
                print(classification_report(testy.argmax(axis=1), pred1_testy.argmax(axis=1)))
                print('** See f1 score for accuracy and class accuracy\n')
                print('Confusion matrix: ')
                print(confusion_matrix(testy.argmax(axis=1), pred1_testy.argmax(axis=1)))

                # plt.plot(history2.epoch, history2.history['loss'], 'g')
                # plt.plot(history2.epoch, history2.history['val_loss'], 'r')
                print()
                print('MSE (if testing on error function) / ReLU (if testing on hidden units)')
                print('Result on train data: ')
                _, acc2_train = model2.evaluate(trainx, trainy)
                pred2_trainy = model2.predict(trainx)
                print(classification_report(trainy.argmax(axis=1), pred2_trainy.argmax(axis=1)))
                print('** See f1 score for accuracy and class accuracy\n')
                print('Confusion matrix: ')
                print(confusion_matrix(trainy.argmax(axis=1), pred1_trainy.argmax(axis=1)))
                print('Result on test data: ')
                _, acc2_test = model2.evaluate(testx, testy)
                pred2_testy = model2.predict(testx)
                print(classification_report(testy.argmax(axis=1), pred2_testy.argmax(axis=1)))
                print('** See f1 score for accuracy and class accuracy\n')
                print('Confusion matrix: ')
                print(confusion_matrix(testy.argmax(axis=1), pred2_testy.argmax(axis=1)))
                print('--------------------------------------------------------------------------------')

        ##
        #   CNN
        elif question == "2":
            # reshape data
            #   to np array
            testx = np.array(test_X)
            trainx = np.array(train_X)
            #   normalize data
            trainx, testx = normalize_data("max", trainx, testx)
            #   reshape
            trainx = trainx.reshape(-1, 8, 8, 1)
            testx = testx.reshape(-1, 8, 8, 1)

            print('Default hyperparameter:\n<1>learning rate = ' + str(lr_d) + '\n<2>momentum = ' + str(mt_d))
            print('<3>training epoch = ' + str(ep_d) + '\n<4>hidden layers (<3) = ' + str(hidl_d))
            print('<5>hidden size = 64\nhidden units = ReLU\nerror function = cross entropy\n')
            lr, mt, ep, hidl, hids, hidu, lf = user_param(hids=64)

            #   make model
            model = tf.keras.Sequential()
            #   input layer
            model.add(keras.layers.Conv2D(32, 3, 3, input_shape=trainx[0].shape, activation=tf.nn.relu))
            #   add layers
            for i in range(hidl):
                model.add(keras.layers.Conv2D(hids, kernel_size=(hidl-i, hidl-i), activation=tf.nn.relu))
                # model.add(keras.layers.MaxPooling2D(2, 2))
            #   output layer
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
            model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt), metrics=['accuracy'])
            history = model.fit(trainx, trainy, epochs=ep, validation_split=0.2, verbose=v)

            # plt.plot(history.epoch, history.history['loss'], 'g')
            # plt.plot(history.epoch, history.history['val_loss'], 'r')
            print()
            print('CNN')
            print('Result on train data: ')
            _, acc_train = model.evaluate(trainx, trainy)
            pred_trainy = model.predict(trainx)
            print(classification_report(trainy.argmax(axis=1), pred_trainy.argmax(axis=1)))
            print('** See f1 score for accuracy and class accuracy')
            print('Confusion matrix: ')
            print(confusion_matrix(trainy.argmax(axis=1), pred_trainy.argmax(axis=1)))
            print('Result on test data: ')
            _, acc_test = model.evaluate(testx, testy)
            pred_testy = model.predict(testx)
            print(classification_report(testy.argmax(axis=1), pred_testy.argmax(axis=1)))
            print('** See f1 score for accuracy and class accuracy\n')
            print('Confusion matrix: ')
            print(confusion_matrix(testy.argmax(axis=1), pred_testy.argmax(axis=1)))
        else:
            break
