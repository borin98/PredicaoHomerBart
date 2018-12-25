import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout,Activation
from keras.utils import np_utils
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.layers.normalization import BatchNormalization
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder

def montaRede ( tamEntrada ) :

    cNN = Sequential ( )

    cNN.add ( Dense(
        units = 3,
        activation = "relu",
        input_dim = tamEntrada
    ) )

    cNN.add ( Dense(
        units = 3,
        activation = "relu",
    ) )

    cNN.add ( Dropout ( 0.1 ) )

    cNN.add ( Dense(
        units = 1
    ) )

    cNN.add ( Activation(
        tf.nn.sigmoid
    ) )

    cNN.compile(
        optimizer = "SGD",
        loss = "binary_crossentropy",
        metrics = ["acc"]
    )

    return cNN

def show_accuracy_chart(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label = 'acc')
    plt.plot(epochs, val_acc, 'b', label="val acc")
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def main (  ) :

    dataset = pd.read_csv ( "personagens.csv" )

    tam = len( dataset.columns ) - 1

    x = dataset.iloc[:, 0:tam ].values
    y = dataset.iloc[:, 6].values

    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform ( y )

    X_train, X_test, y_train, y_test = train_test_split(
     x,
     y,
     test_size = 0.25
    )

    cNN = montaRede ( tamEntrada = tam )

    sequential_model_to_ascii_printout ( cNN )

    dadosTreinamento = cNN.fit(
        X_train,
        y_train,
        batch_size = 10,
        epochs = 2000,
        validation_data = ( X_test, y_test )
    )

    resultado = cNN.evaluate (
        X_test,
        y_test
    )

    param_range = np.logspace(-6, -1, 5)

    show_accuracy_chart ( dadosTreinamento )

    print("Média de acertos : {} %".format( resultado[1]*100 ))

    vizualizaResultados ( train_scores, test_scores, param_range)

main()
