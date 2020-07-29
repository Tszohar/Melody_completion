from datetime import datetime
import os

import tensorflow as tf
import numpy as np
from config import Config
from dataset import MidiDataset
from model import get_model
from train import get_data

if __name__ == "__main__":
    filepath = "/home/guy/melody_completions/runs/run_20200613_215938/model.33-0.14.h5"
    x_train, y_train, x_test, y_test = get_data(base_folder=Config().BASE_FOLDER)

    model = get_model()

    model.load_weights(filepath)

    y_pred = model.predict(x_train[:1])