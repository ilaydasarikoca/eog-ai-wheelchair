"""Hardware-free inference smoke check; not an accuracy evaluation."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from eog_ortak import CLASSES, hazirla

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--samples', type=int, default=363)
    parser.add_argument('--model', type=Path,
        default=Path(__file__).resolve().parents[1] / 'models/model_eog.pkl')
    args = parser.parse_args()
    if args.start < 0 or args.samples < 10:
        parser.error('start must be non-negative and samples must be at least 10')
    signal = np.loadtxt(args.csv)
    if signal.ndim != 1:
        parser.error('Expected a one-column signal CSV')
    vector = hazirla(signal[args.start:args.start + args.samples])
    if vector is None:
        parser.error('Invalid signal window')
    model = joblib.load(args.model)
    probabilities = model.predict_proba(vector.reshape(1, -1))[0]
    label = int(model.predict(vector.reshape(1, -1))[0])
    print(json.dumps({'prediction': CLASSES[label],
        'probabilities': {CLASSES[int(k)]: float(v) for k, v in zip(model.classes_, probabilities)},
        'input_features': len(vector),
        'purpose': 'Model smoke check only; arbitrary window is not a benchmark'}, indent=2))

if __name__ == '__main__':
    main()
