import json
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Flatten, Dense, BatchNormalization, LayerNormalization
from sklearn.model_selection import train_test_split

from config import Config


def get_model():
    """
    # The input was entered into an
    # LSTM layer with 512 nodes. Next, dropout regularization was used in order to reduce overfitting
    # [14], after which there was another LSTM layer with 512 nodes. A Dense layer then lowered the
    # number of nodes to 128 (corresponding to 128 possible pitches), which was sent to the output.
    # This structure was determined through experimentation. Mean squared error was used as loss
    # function for training, with linear activation. RMSProp optimization [12] was used to speed up
    # the training of the network.
    :return: Note Predictor model
    """
    lstm_nodes = 128
    dropout_rate = 0.0

    model = Sequential()
    model.add(tf.keras.layers.Input((Config().NUM_NOTES - 1, 66)))
    # model.add(tf.keras.layers.Input((Config().NUM_NOTES - 1, 128)))
    # model.add(Dense(units=64, activation='relu', use_bias=False))
    # model.add(LSTM(units=lstm_nodes,
    #                recurrent_dropout=dropout_rate,
    #                return_sequences=True))
    # model.add(LayerNormalization())
    model.add(LSTM(units=lstm_nodes))
    model.add(LayerNormalization())
    model.add(Dense(66, activation='sigmoid'))

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    model.summary()

    return model


