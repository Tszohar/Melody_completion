from datetime import datetime

import functools
import os

import tensorflow as tf
import numpy as np
from config import Config
from dataset import MidiDataset
from model import get_model


def concat_datasets(base_folder, folders):
    x_list = []
    y_list = []
    dataset = None
    samples_num = 0
    datasets = []
    data_gens = []
    for folder in folders:
        folder_ = os.path.join(base_folder, folder)
        print("\rProcessing folder: {}".format(folder_))
        # x, y = MidiDataset(folder_).get_data()
        midi_dataset = MidiDataset(folder_)

        for song_idx in range(len(midi_dataset.files_data)):
            samples_num += len(tf.keras.preprocessing.sequence.TimeseriesGenerator(data=midi_dataset.files_data[song_idx],
                                                                                   targets=midi_dataset.files_data[song_idx],
                                                                                   length=Config().NUM_NOTES-1,
                                                                                   batch_size=1))
            data_gen = functools.partial(tf.keras.preprocessing.sequence.TimeseriesGenerator,
                                         data=midi_dataset.files_data[song_idx],
                                         targets=midi_dataset.files_data[song_idx],
                                         length=Config().NUM_NOTES-1,
                                         batch_size=1)
            data_gens.append(data_gen)
            dataset_ = tf.data.Dataset.from_generator(generator=data_gen,
                                                      output_types=(tf.float32, tf.float32),
                                                      output_shapes=((1, Config().NUM_NOTES-1, 128), (1, 128)))
            datasets.append(dataset_)
            if not dataset:
                dataset = dataset_
            else:
                dataset = dataset.concatenate(dataset_)

    dataset = dataset.unbatch().shuffle(buffer_size=10000).batch(Config().BATCH_SIZE, drop_remainder=False)
    # dataset = dataset.prefetch(10)
    return dataset


def get_data(base_folder: str):
    print("Generating train dataset")
    train_dataset = concat_datasets(base_folder, Config().TRAIN_FOLDERS)
    print("Generating test dataset")
    test_dataset = concat_datasets(base_folder, Config().TEST_FOLDERS)

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = get_data(base_folder=Config().BASE_FOLDER)

    model = get_model()

    timestamp_str = datetime.strftime(datetime.now(), format="%Y%m%d_%H%M%S")
    run_path = "./runs/run_{}".format(timestamp_str)
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    my_callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(run_path, 'model.{epoch:02d}.h5')),
        # tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_path, 'logs')),
    ]
    model.fit(train_dataset, epochs=100, validation_data=test_dataset, validation_steps=1000,
              callbacks=my_callbacks)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(test_dataset)
    print('test loss, test acc:', results)
