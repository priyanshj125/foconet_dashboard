import torch
import numpy as np
import sys
import os
import glob

sys.path.append('.')
import config
from data_loader import WaveFormDataset
from torch.utils.data import DataLoader
from model import FocoNet
from pyrocko import moment_tensor as pmt

def main():
    # 1. Setup device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")

    # We will get the folder name directly from dataset.eids

    # 3. Setup Data
    dataset = WaveFormDataset('dev')
    dataloader = DataLoader(dataset, batch_size=1)

    # 3. Load Model
    model = FocoNet().to(device)
    model_path = 'model/FocoNet_Full.pth'
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found!")
        # let's try another model if available
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 4. Predict
    print("\n--- Predictions ---\n")
    results = []
    
    for i, data in enumerate(dataloader):
        polarities = data['wave'].to(device)
        sta_param = data['sta_param'].to(device)
        sta_num = data['sta_num'].to(device)
        sta_mask = data['sta_mask'].to(device)

        with torch.no_grad():
            outputs = model(polarities, sta_param, sta_num, sta_mask, mode='test')
            
        outputs_np = outputs.cpu().numpy()[0]
        p_axis = outputs_np[0]
        t_axis = outputs_np[1]
        
        try:
            # Normalize vectors first
            p_axis = p_axis / np.linalg.norm(p_axis)
            t_axis = t_axis / np.linalg.norm(t_axis)
            strike, dip, rake = pmt.pt_axes_to_strike_dip_rake(p_axis, t_axis)
            results.append({
                'event_index': i + 1,
                'folder_name': dataset.eids[i] if hasattr(dataset, 'eids') else 'unknown',
                'strike': strike,
                'dip': dip,
                'rake': rake
            })
            if (i+1) % 10 == 0:
                print(f"Processed {i+1} events...")
        except Exception as e:
            print(f"Event {i+1}: Conversion failed: {e}")
            
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)
    print("\n✅ Saved all predictions to FocoNet_Full/predictions.csv")

if __name__ == "__main__":
    main()
