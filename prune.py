import pandas as pd
import numpy as np

data = pd.read_csv("./data/processed(merged version).csv", index_col=0)
print("original datashape: " + str(data.shape))
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
    "Hybrid slot",
    "Infrared port",
    "Multi SIM mode",
    "NFC*",
    "Number of SIM*",
    "Stabilization",
    "Full charging time (min)",
    "Touch sampling rate",
    "Total score",
    "Speakers_Mono",
    "Speakers_Stereo"
]

optionalLines = [
    "Height (mm)",
    "Battery capacity (mAh)",
    "Flash",
    "Manufacturing",
    "Neural processor (NPU)",
    "Proximity sensor",
    "Rear material"
]
data = data.drop(columns=removedColumns)

print("pruned.csv: " + str(data.shape))
data.to_csv("./data/pruned.csv", index=0)

data = data.drop(columns=optionalLines)
print("aggr_pruned.csv: " + str(data.shape))
data.to_csv("./data/aggr_pruned.csv", index=0)

