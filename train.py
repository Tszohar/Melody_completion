from datetime import datetime
import os

import tensorflow as tf
import numpy as np
from config import Config
from dataset import MidiDataset
from model import get_model


def concat_datasets(base_folder, folders):
    x_list = []
    y_list = []
    for folder in folders:
        folder_ = os.path.join(base_folder, folder)
        print("\rProcessing folder: {}".format(folder_))
        x, y = MidiDataset(folder_).get_data()
        x_list.append(x)
        y_list.append(y)

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


def get_data(base_folder: str):
    print("Generating train dataset")
    x_train, y_train = concat_datasets(base_folder, Config().TRAIN_FOLDERS)
    print("Generating test dataset")
    x_test, y_test = concat_datasets(base_folder, Config().TEST_FOLDERS)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data(base_folder=Config().BASE_FOLDER)

    model = get_model()

    batch_size = 128
    timestamp_str = datetime.strftime(datetime.now(), format="%Y%m%d_%H%M%S")
    run_path = "./runs/run_{}".format(timestamp_str)
    my_callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(run_path, 'model.{epoch:02d}-{val_loss:.2f}.h5')),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_path, 'logs')),
    ]
    model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=batch_size, callbacks=my_callbacks)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test loss, test acc:', results)
