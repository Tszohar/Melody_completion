from datetime import datetime
import os

import tensorflow as tf
import numpy as np
from config import Config
from dataset import MidiDataset
from model import get_model
from train import get_data


def load_model_by_path(model_path):
    model = get_model()
    model.load_weights(model_path)
    return model


if __name__ == "__main__":
    # filepath = "/home/guy/melody_completions/runs/run_20200613_215938/model.33-0.14.h5"
    # filepath = '/home/tsofit/OneDrive/music/Melody_completions/trained_models/model.01.h5'
    filepath = '/home/tsofit/OneDrive/music/Melody_completions/trained_models/model.22.h5'
    # x_train, y_train, x_test, y_test = get_data(base_folder=Config().BASE_FOLDER)
    train_dataset, test_dataset = get_data(base_folder='/home/tsofit/maestro_dataset/maestro-v2.0.1')

    model = get_model()

    model.load_weights(filepath)
    it = iter(train_dataset)
    data = it.next()
    y_pred = model.predict(data[0])
    print('bla')