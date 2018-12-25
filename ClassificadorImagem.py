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
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, image

def vizualizaResultados ( cnn, epocas ) :

    plt.figure(0)
    plt.plot(cnn.history['acc'],'r')
    plt.plot(cnn.history['val_acc'],'g')
    plt.xticks(np.arange(0, ( epocas + 1 ), 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])
    plt.grid(True)

    plt.show()

    plt.figure(0)
    plt.plot(cnn.history['loss'],'r')
    plt.plot(cnn.history['val_loss'],'g')
    plt.xticks(np.arange(0, (epocas + 1), 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
    plt.grid(True)

    plt.show()

    return

def montaCNN ( largImagem, alturaImagem ) :

    print("---------- Inicio de Treinamento ------------\n")

    cNN = Sequential ( )

    # primeira camadas convolução
    cNN.add ( Conv2D(
        filters = 16,
        kernel_size = (3, 3),
        input_shape = ( largImagem, alturaImagem, 3 ),
        activation = "relu"
    ) )
    cNN.add ( BatchNormalization() )
    cNN.add(MaxPooling2D(
        pool_size = (2, 2)
    ))

    # segunda camadas convolução
    cNN.add ( Conv2D(
        filters = 16,
        kernel_size = (3, 3),
        activation = "relu"
    ) )
    cNN.add ( BatchNormalization() )
    cNN.add(MaxPooling2D(
        pool_size = (2, 2)
    ))
    #cNN.add ( BatchNormalization() )

    """cNN.add( Conv2D(
        filters = 128,
        kernel_size = (3, 3),
        activation = "relu"
    ))"""

    cNN.add ( BatchNormalization() )
    # camada de Flatten
    cNN.add ( Flatten() )

    # rede neural densa
    cNN.add ( Dense(
        units = 4,
        activation = "relu"
    ) )
    cNN.add ( Dropout ( 0.2 ) )
    cNN.add(Dense(
        units = 4,
        activation = "relu"
    ))
    #cNN.add ( Dropout ( 0.5 ) )
    """cNN.add(Dropout ( 0.5 ) )
    cNN.add(Dense(
        units = 32,
        activation = "relu"
    ))"""

    cNN.add(Dense(
        units = 1
    ))
    cNN.add ( Activation (
        tf.nn.sigmoid
    ) )

    cNN.compile (
         loss = "binary_crossentropy",
         optimizer = "adam",
         metrics = ["accuracy"]
     )

    return cNN

def main (  ) :

    epocas = int (input("Digite a quantidade de épocas que treinará : " ) )

    geradorTreinamento = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 7,
        horizontal_flip = True,
        shear_range = 0.2,
        height_shift_range = 0.07,
        zoom_range = 0.2
    )

    geradorTeste = ImageDataGenerator(
        rescale = 1./255
    )

    baseTreinamento = geradorTreinamento.flow_from_directory(
        "dataset_personagens/training_set",
        target_size = ( 64, 64 ),
        batch_size = 64,
        class_mode = "binary"
    )

    baseTeste = geradorTreinamento.flow_from_directory(
        "dataset_personagens/test_set",
        target_size = ( 64, 64 ),
        batch_size = 64,
        class_mode = "binary"
    )

    cNN = montaCNN (
        alturaImagem = 64,
        largImagem = 64
    )

    sequential_model_to_ascii_printout ( cNN )

    """

    Melhores parâmetros
    epocas = 5,
     steps_per_epoch = 1000,
     validation_steps = 120
    """

    avaliacao = cNN.fit_generator ( baseTreinamento,
              steps_per_epoch = 125,
              epochs = epocas,
              validation_data = baseTeste,
              validation_steps = 63
    )

    vizualizaResultados ( avaliacao, epocas )

if __name__ == '__main__':
    main()
