import os
import pandas as pd
import numpy as np
from mido import MidiFile

FOLDER = 'midi_files'
NUM_NOTES = 1000


def read_midi_files(folder: str) -> np:
    """
    This function reads each midi file in folder and return a numpy array of midi info
    :param folder: midi folder path
    :return: numpy array of midi information
    """
    files_list = os.listdir(folder)
    midies = []
    for file in files_list:
        filename = os.path.join(folder, file)
        mid = MidiFile(filename)
        df = extract_track_info(mid)
        df = preprocess_df(df, filename)
        # Convert midi file info to Numpy
        mid_np = df[['note', 'velocity', 'time_elapsed']].iloc[:NUM_NOTES].to_numpy()
        midies.append(mid_np.reshape(1, mid_np.shape[0], mid_np.shape[1]))
    return np.array(midies)


def preprocess_df(df: pd.DataFrame, filename: str) -> pd:
    """
    This function pre process single midi information as df
    :param df: midi information
    :param filename: midi file name
    :return: processed df
    """
    # Filter only note_on rows with velocity greater than 0
    df = df[df['message_type'] == 'note_on']
    df = df.loc[df['velocity'] != '0']

    # Transform the time and note attributes from strings to floats
    df['time'] = df.time.astype(float)
    df['note'] = df.note.astype(int)

    # Engineer a time elapsed attribute equal to the cumulative sum of time.
    df['time_elapsed'] = df.time.cumsum()

    df = df.drop(['value', 'control'], axis=1)
    df['song'] = filename
    return df


def extract_track_info(mid: MidiFile) -> pd:
    """
    This function extract the midi information from MidFile object
    :param mid: MidFile object of a single midi file
    :return: pd DataFrame of midi information
    """
    message_list = [str(msg) for msg in mid.tracks[1][1:-1]]
    message_strings_split = [msg.split() for msg in message_list]
    message_type = [item[0] for item in message_strings_split]
    attributes = [item[1:] for item in message_strings_split]

    attributes_list = [{}]
    for item in attributes:
        for i in item:
            key, val = i.split("=")
            if key in attributes_list[-1]:
                attributes_list.append({})
            attributes_list[-1][key] = val

    df = pd.DataFrame(message_type)
    df_attrs = pd.DataFrame.from_dict(attributes_list)
    df = pd.concat([df, df_attrs], axis=1)
    columns = ['message_type', 'channel', 'control', 'value', 'time', 'note', 'velocity']
    df.columns = columns

    return df


def main():
    midies = read_midi_files(FOLDER)
    print(midies)


if __name__ == '__main__':
    main()
