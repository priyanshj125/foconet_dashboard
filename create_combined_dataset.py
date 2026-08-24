import glob
import os
import numpy as np

XQ_DIR = '/home/priyansh/Desktop/foco/FocoNet/XQ_npy '
npy_files = sorted(glob.glob(os.path.join(XQ_DIR, '*.npy')))

combined_data = {}
for f in npy_files:
    if os.path.basename(f) == 'XQ_dataset.npy': continue
    event_id = os.path.basename(f).replace('.npy', '')
    data = np.load(f, allow_pickle=True).item()
    combined_data[event_id] = data

np.save(os.path.join(XQ_DIR, 'XQ_dataset.npy'), combined_data)
print(f"Combined {len(combined_data)} events into XQ_dataset.npy")
