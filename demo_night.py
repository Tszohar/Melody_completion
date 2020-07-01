from dataset import MidiDataset
from predict import load_model_by_path
import numpy as np

folder = "../maestro_dataset/demo_midi"

mid = MidiDataset(folder)

sample_number = 0
midi_sample = mid.files_data[sample_number]

dim = midi_sample.shape[0]
midi_sample_first = midi_sample[:-128, :]
midi_sample_last = midi_sample[dim-128:, :]

model_path = '/home/guy/melody_completions/runs/run_20200630_234617/model.16.h5'
model = load_model_by_path(model_path)
y_pred = model.predict(midi_sample_last[None, ::])

# Convert to 128 Notes
full_midi_sample_first = np.zeros((midi_sample_first.shape[0], 128))
full_midi_sample_first[:, 28:94] = midi_sample_first
full_midi_sample_last = np.zeros((midi_sample_last.shape[0], 128))
full_midi_sample_last[:, 28:94] = midi_sample_last

space = np.zeros((1, 128))
prediction = np.zeros((1, 128))
prediction[:, 28:94] = y_pred

full_midi = np.concatenate((full_midi_sample_first, full_midi_sample_last, space, prediction), axis=0)

import pickle
with open('predicted_midi_with_space.pkl', 'wb') as f:
    pickle.dump(full_midi, f)

print('bla')