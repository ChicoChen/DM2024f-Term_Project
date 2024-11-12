import pandas as pd
import numpy as np

data = pd.read_csv("./data/processed(merged version).csv")
removedColumns = [
    "Accelerometer",
    "Ambient light sensor",
    "Autofocus_Phase autofocus (Dual Pixel)",
    "Autofocus_Laser autofocus",
    "Battery replaceability",
    "Battery type",
    "Bundled charger",
    "Depth sensor (TOF 3D)",
    "Face recognition sensor",
    "Fast charging",
    "Fingerprint",
    "Height (mm)",
    "Hybrid slot",
    "Infrared port",
    "Multi SIM mode",
    "NFC*",
    "Number of SIM*",
    "Stabilization",
    "Full charging time (min)",
    "Touch sampling rate"
]

data = data.drop(columns=removedColumns)
print(data.shape)
data.to_csv("./data/pruned.csv")
