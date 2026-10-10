"""Train the architecture observed in the supplied model, using separate event CSVs."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eog_ortak import CLASSES, hazirla

FOLDERS = {'NOTR': 'notr', 'SAG': 'sag', 'SOL': 'sol', 'GOZ_KIRPMA': 'kirpma'}

def make_model():
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(96, 48), max_iter=1800, random_state=42))

def load_events(base):
    X, y, manifest, skipped = [], [], [], []
    for folder, label in FOLDERS.items():
        for path in sorted((base / folder).rglob('*.csv')):
            try:
                values = np.loadtxt(path)
                if values.ndim != 1:
                    raise ValueError('Expected a one-column event signal')
                vector = hazirla(values)
                if vector is None:
                    raise ValueError('Signal is too short or contains non-finite values')
            except (OSError, ValueError) as exc:
                skipped.append({'file': str(path.relative_to(base)), 'reason': str(exc)})
                continue
            X.append(vector)
            y.append(CLASSES.index(label))
            manifest.append(str(path.relative_to(base)))
    return np.asarray(X), np.asarray(y), manifest, skipped

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/retrained_model.pkl'))
    args = parser.parse_args()
    X, y, manifest, skipped = load_events(args.dataset)
    counts = np.bincount(y.astype(int), minlength=4) if len(y) else np.zeros(4, dtype=int)
    if any(counts < 5):
        parser.error('Provide at least five valid separate events in each of NOTR, SAG, SOL, GOZ_KIRPMA. '
                     'Included data/raw streams are not this dataset. Counts: ' + str(dict(zip(CLASSES, counts.tolist()))))
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    model = make_model()
    model.fit(X[train], y[train])
    predicted = model.predict(X[test])
    accuracy = float(accuracy_score(y[test], predicted))
    metrics = {
        'accuracy': accuracy, 'train_events': len(train), 'test_events': len(test),
        'class_order': CLASSES, 'class_counts': dict(zip(CLASSES, counts.tolist())),
        'confusion_matrix': confusion_matrix(y[test], predicted, labels=range(4)).tolist(),
        'classification_report': classification_report(y[test], predicted, labels=range(4),
            target_names=CLASSES, output_dict=True, zero_division=0),
        'split': 'Stratified 80/20 sample split, seed 42; not participant-independent',
        'train_files': [manifest[i] for i in train], 'test_files': [manifest[i] for i in test],
        'skipped_files': skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    supplied = Path(__file__).resolve().parents[1] / 'models/model_eog.pkl'
    if args.output.resolve() == supplied.resolve():
        parser.error('Choose a new output path; the supplied model is preserved.')
    joblib.dump(model, args.output)
    args.output.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Test accuracy on this dataset: {accuracy:.1%}')
    print('Class order:', ', '.join(CLASSES))
    print(np.asarray(metrics['confusion_matrix']))
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
