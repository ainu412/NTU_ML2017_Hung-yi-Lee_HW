import os

from matplotlib.ticker import MultipleLocator
from utility import clean_data
from matplotlib import pyplot as plt
import pickle
import numpy as np
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
import tensorflow.python.keras.callbacks as cb
from tensorflow.python.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight

load_from_file = True
epoch_num = 100

def main():
    name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    train_data = clean_data('data/train.csv')

    train = train_data.feature.reshape((-1, 48, 48, 1))/255
    train_x = train[:-2000]
    train_onehot = train_data.onehot[:-2000]
    valid_x = train[-2000:]
    valid_onehot = train_data.onehot[-2000:]

    class_weight = compute_class_weight(class_weight='balanced',
                                        classes=np.unique(train_data.label),
                                        y=train_data.label)
    def cnn():
        #CNN model

        # define architecture
        inputs = Input(shape=(48,48,1))

        ## First convolutional layer with ReLU-activation and max-pooling.
        net = Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
                     activation='relu', name='layer_conv1')(inputs)
        net = MaxPooling2D(pool_size=2, strides=2)(net)
        net = BatchNormalization(axis = -1)(net)
        net = Dropout(0.25)(net)

        ## Second convolutional layer with ReLU-activation and max-pooling.
        net = Conv2D(kernel_size=5, strides=1, filters=128, padding='same',
                     activation='relu', name='layer_conv2')(net)
        net = MaxPooling2D(pool_size=2, strides=2)(net)
        net = BatchNormalization(axis = -1)(net)
        net = Dropout(0.25)(net)

        ## Third convolutional layer with ReLU-activation and max-pooling.
        net = Conv2D(kernel_size=5, strides=1, filters=256, padding='same',
                     activation='relu', name='layer_conv3')(net)
        net = MaxPooling2D(pool_size=2, strides=2)(net)
        net = BatchNormalization(axis = -1)(net)
        net = Dropout(0.5)(net)

        ## Flatten the output of the conv-layer from 4-dim to 2-dim.
        net = Flatten()(net)

        ## First fully-connected / dense layer with ReLU-activation.
        net = Dense(128)(net)
        net = BatchNormalization(axis = -1)(net)
        net = Activation('relu')(net)

        ## Last fully-connected / dense layer with softmax-activation, so it can be used for classification.
        net = Dense(7)(net)
        net = BatchNormalization(axis = -1)(net)
        net = Activation('softmax')(net)
        ## Output of the Neural Network.
        outputs = net
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='Adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        if load_from_file:
            model.load_weights('model/CNN-best_weights.hdf5')
            with open('train-history/CNN-train-history.txt', 'rb') as file_txt:
                history = pickle.load(file_txt)

            # summarize history for accuracy
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('cnn accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            a = history['val_acc']
            max_index = np.argmax(a)
            plt.plot(max_index, a[max_index], 'ro')
            show_max = '[' + str(max_index) + ' ' + ('%.2f' % a[max_index]) + ']'
            plt.annotate(show_max, xytext=(max_index, a[max_index]), xy=(max_index, a[max_index]))
            plt.savefig('img/report/CNN-TrainProcess-accuracy.png')
            plt.show()
            # summarize history for loss
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('cnn loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            a = history['val_loss']
            min_index = np.argmin(a)
            plt.plot(min_index, a[min_index], 'ro')
            show_max = '[' + str(min_index) + ' ' + ('%.2f' % a[min_index]) + ']'
            plt.annotate(show_max, xytext=(min_index, a[min_index]), xy=(min_index, a[min_index]))
            plt.savefig('img/report/CNN-TrainProcess-loss.png')
            plt.show()
        else:
            # tensorboard = TensorBoard(log_dir="logs\{0}".format(time())) no early stopping
            best_weights_filepath = 'model/CNN-best_weights.hdf5'
            save_best = cb.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,
                                                       save_best_only=True)
            early_stopping = cb.EarlyStopping(monitor='val_loss', patience=8)
            history = model.fit(x=train_x,
                                y=train_onehot,
                                validation_data=(valid_x, valid_onehot),
                                class_weight=class_weight,
                                epochs=epoch_num,
                                batch_size=64,
                                callbacks=[save_best, early_stopping],
                                )
            with open('train-history/CNN-train-history.txt', 'wb') as file_txt:
                pickle.dump(history.history, file_txt)

    cnn()
    #DNN model
    def dnn():
        inputs = Input(shape=(48,48,1))

        dnn = Flatten()(inputs)

        dnn = Dense(512)(dnn)
        dnn = BatchNormalization(axis = -1)(dnn)
        dnn = Activation('relu')(dnn)
        dnn = Dropout(0.25)(dnn)

        dnn = Dense(1024)(dnn)
        dnn = BatchNormalization(axis = -1)(dnn)
        dnn = Activation('relu')(dnn)
        dnn = Dropout(0.5)(dnn)

        dnn = Dense(512)(dnn)
        dnn = BatchNormalization(axis = -1)(dnn)
        dnn = Activation('relu')(dnn)
        dnn = Dropout(0.5)(dnn)

        dnn = Dense(7)(dnn)
        dnn = BatchNormalization(axis = -1)(dnn)
        dnn = Activation('softmax')(dnn)

        outputs = dnn

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='Adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        if load_from_file:
            model.load_weights('model/DNN-best_weights.hdf5')
            with open('train-history/DNN-train-history.txt', 'rb') as file_txt:
                history = pickle.load(file_txt)

            # summarize history for accuracy
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('dnn accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            a = history['val_acc']
            max_index = np.argmax(a)
            plt.plot(max_index, a[max_index], 'ro')
            show_max = '[' + str(max_index) + ' ' + ('%.2f' % a[max_index]) + ']'
            plt.annotate(show_max, xytext=(max_index, a[max_index]), xy=(max_index, a[max_index]))
            plt.savefig('img/report/DNN-TrainProcess-accuracy.png')
            plt.show()
            # summarize history for loss
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('dnn loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            a = history['val_loss']
            min_index = np.argmin(a)
            plt.plot(min_index, a[min_index], 'ro')
            show_max = '[' + str(min_index) + ' ' + ('%.2f' % a[min_index]) + ']'
            plt.annotate(show_max, xytext=(min_index, a[min_index]), xy=(min_index, a[min_index]))
            plt.savefig('img/report/DNN-TrainProcess-loss.png')
            plt.show()
        else:
            # tensorboard = TensorBoard(log_dir="logs\{0}".format(time())) no early stopping
            best_weights_filepath = 'model/DNN-best_weights.hdf5'
            save_best = cb.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,
                                                       save_best_only=True)
            early_stopping = cb.EarlyStopping(monitor='val_loss', patience=8)
            history = model.fit(x=train_x,
                                 y=train_onehot,
                                 validation_data=(valid_x, valid_onehot),
                                 class_weight=class_weight,
                                 epochs=epoch_num, batch_size=64,
                                 callbacks=[save_best, early_stopping]
                                 )
            with open('train-history/DNN-train-history.txt', 'wb') as file_txt:
                pickle.dump(history.history, file_txt)
    dnn()


if __name__ == '__main__':
    main()


