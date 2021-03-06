import numpy as np

def import_results():
    cnn_results = np.array([
    [np.nan,	95.1,	99.3,	94.1,	99.3,	98.8,	98.6,	98.8,	96.5,	95.2],
    [95.1,	np.nan,	95.7,	94.5,	89.9,	78.2,	92.4,	99.0,	77.0,	86.6],
    [99.3,	95.7,	np.nan,	99.9,	95.6,	96.1,	87.7,	96.9,	99.5,	99.1],
    [94.1,	94.5,	99.9,	np.nan,	98.5,	96.8,	99.9,	99.9,	97.5,	94.1],
    [99.3,	89.9,	95.6,	98.5,	np.nan,	90.1,	89.7,	98.8,	87.5,	83.3],
    [98.8,	78.2,	96.1,	96.8,	90.1,	np.nan,	90.8,	95.3,	91.9,	92.8],
    [98.6,	92.4,	87.7,	99.9,	89.7,	90.8,	np.nan,	93.0,	87.5,	93.5],
    [98.8,	99.0,	96.9,	99.9,	98.8,	95.3,	93.0,	np.nan,	98.2,	96.9],
    [96.5,	77.0,	99.5,	97.5,	87.5,	91.9,	87.5,	98.2,	np.nan,	73.7],
    [95.2,	86.6,	99.1,	94.1,	83.3,	92.8,	93.5,	96.9,	73.7,	np.nan]])

    qnn4eo_results = np.array([
    [np.nan,	95.5,	99.8,	93.5,	99.3,	98.3,	99.7,	99.7,	97.8,	95.3],
    [95.5,	np.nan,	99.5,	95.3,	93.4,	85.4,	94.6,	98.6,	82.7,	87.5],
    [99.8,	99.5,	np.nan,	99.9,	94.7,	97.4,	86.6,	97.9,	99.3,	97.9],
    [93.5,	95.3,	99.9,	np.nan,	98.7,	97.5,	99.9,	99.9,	98.2,	97.4],
    [99.3,	93.4,	94.7,	98.7,	np.nan,	91.6,	91.8,	91.9,	87.4,	94.3],
    [98.3,	85.4,	97.4,	97.5,	91.6,	np.nan,	90.2,	96.1,	93.9,	92.2],
    [99.7,	94.6,	86.6,	99.9,	91.8,	90.2,	np.nan,	95.1,	89.4,	96.0],
    [99.7,	98.6,	97.9,	99.9,	91.9,	96.1,	95.1,	np.nan,	98.9,	98.0],
    [97.8,	82.7,	99.3,	98.2,	87.4,	93.9,	89.4,	98.9,	np.nan,	75.0],
    [95.3,	87.5,	97.9,	97.4,	94.3,	92.2,	96.0,	98.0,	75.0,	np.nan]])

    classes_name = np.array(["Residential", "Highway", "SeaLake", "Industrial", "Annual\nCrop", "River", "Pasture",	"Forest", "Permanent\nCrop", "Herbaceous\nVegetation"])

    return cnn_results, qnn4eo_results, classes_name