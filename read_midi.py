import os
import pickle
import shutil
import pandas as pd
import numpy as np
from mido import MidiFile
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config


def read_midi_files(folder: str) -> np:
    """
    This function reads each midi file in folder and return a numpy array of midi info
    :param folder: midi folder path
    :return: numpy array of midi information and numpy target array
    """
    files_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files_list = sorted(list(filter(lambda file_name: file_name.endswith(".midi"), files_list)))

    dst_dir = os.path.join(folder, Config().PROCESSED_FOLDER)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for file in tqdm(files_list):
        dst_file = os.path.join(dst_dir, os.path.splitext(file)[0]) + '.pkl'

        # Skip file if processed file already exists
        if os.path.isfile(dst_file):
            continue

        filename = os.path.join(folder, file)
        mid = MidiFile(filename)
        df = extract_track_info(mid)

        df = preprocess_df(df, filename)
        midi_arr = convert_to_matrix(df)

        with open(dst_file, 'wb') as f:
            pickle.dump(midi_arr, f)


def preprocess_df(df: pd.DataFrame, filename: str) -> pd:
    """
    This function pre process single midi information as df
    :param df: midi information
    :param filename: midi file name
    :return: processed df
    """
    # Engineer a time elapsed attribute equal to the cumulative sum of time.
    df['time_elapsed'] = df.time.cumsum()
    tick_resolution = 1920 / Config().NOTE_RESOLUTION
    df['time_idx'] = pd.cut(x=df.time_elapsed,
                            bins=np.arange(-1e-3, df.time_elapsed.max() + tick_resolution + 1e-3, tick_resolution),
                            labels=False)
    df['song'] = filename
    return df


def convert_to_matrix(df: pd.DataFrame) -> np.ndarray:
    time_list = df.time_idx.unique()
    try:
        midi_array = np.zeros((time_list.max(), 128))
    except:
        midi_array = np.zeros((time_list.max(), 128))
    df = df.groupby(['time_idx', 'note']).last().reset_index()
    notes_indices = df.note.values
    velocities = df.velocity.values
    velocities[velocities > 0] = 1
    for i in range(midi_array.shape[0]):
        if i > 0:
            midi_array[i] = midi_array[i - 1]
        first_index = np.searchsorted(df.time_idx, i, side='left')
        last_index = np.searchsorted(df.time_idx, i, side='right')
        midi_array[i, notes_indices[first_index:last_index]] = velocities[first_index:last_index]

    return midi_array


def extract_track_info(mid: MidiFile) -> pd.DataFrame:
    """
    This function extract the midi information from MidFile object
    :param mid: MidFile object of a single midi file
    :return: pd DataFrame of midi information
    """
    chosen_track = np.argmax([len(trk) for trk in mid.tracks])
    message_list = [msg for msg in mid.tracks[chosen_track] if msg.type == "note_on"]

    def message_to_dict(msg):
        return {
            "velocity": msg.velocity / 127,
            "time": msg.time,
            "note": msg.note
        }
    df = pd.DataFrame.from_dict(list(map(message_to_dict, message_list)))

    return df


def plot_sheet(midi_array: np.ndarray):
    plt.imshow(midi_array[:Config().NUM_NOTES].T)
    plt.title("Midi Notes")
    plt.xlabel("Tick")
    plt.ylabel("Note")
    plt.show()


def main():
    read_midi_files('../maestro_dataset/demo_midi')


if __name__ == '__main__':
    main()
