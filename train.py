import os

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
    x_train, y_train, x_test, y_test = get_data(base_folder="/home/tsofit/maestro_dataset/maestro-v2.0.0")
    model = get_model()

    batch_size = 1024
    model.fit(x_train, y_train, epochs=100, batch_size=batch_size)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test loss, test acc:', results)
