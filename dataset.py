import glob
import os
import pickle
from typing import List
import numpy as np

from config import Config
from read_midi import read_midi_files


class MidiDataset:
    def __init__(self, folder: str):
        self._processed_folder = os.path.join(folder, Config().PROCESSED_FOLDER)
        # if not os.path.isdir(self._processed_folder):
        self.preprocess_data_folder(folder=folder)
        files_data = self.read_pkls(folder=self._processed_folder)
        self._samples_list, self._data, self._targets = self.split_to_samples(files_data)

    @classmethod
    def preprocess_data_folder(cls, folder: str):
        read_midi_files(folder=folder)

    @classmethod
    def read_pkls(cls, folder: str):
        # Find all serialized midi files
        file_names = glob.glob1(folder, "*.pkl")

        # Go over the files in the processed folder and read them to memory
        files_data = []
        for file_name in file_names:
            with open(os.path.join(folder, file_name), 'rb') as f:
                files_data.append(pickle.load(f))

        return files_data

    def split_to_samples(self, files_data: List[np.ndarray]):
        sample_indices = []
        data_arrays = []
        target_arrays = []
        for idx, file_data in enumerate(files_data):
            start_indices = np.arange(0, file_data.shape[0], Config().NUM_NOTES)
            file_idx = np.full(start_indices.shape[0], idx)
            sample_indices.append(np.stack((file_idx, start_indices), axis=-1))
            for start in start_indices:
                data_array = file_data[start:start + Config().NUM_NOTES]
                if data_array.shape[0] == Config().NUM_NOTES:
                    data_arrays.append(data_array[:-1])
                    target_arrays.append(data_array[-1])
        return np.concatenate(sample_indices, axis=0), np.stack(data_arrays), np.stack(target_arrays)

    def get_data(self):
        return self._data, self._targets


if __name__ == "__main__":
    midi = MidiDataset(Config().FOLDER)
    data = midi.get_data()
    print("bla")