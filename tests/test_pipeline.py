"""Model compatibility and preprocessing tests with no GUI, serial port or motors."""
from pathlib import Path
import sys
import unittest
import tempfile
import os
import runpy
import types
from unittest.mock import patch
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from eog_ortak import CLASSES, hazirla
from train import make_model, load_events

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / 'models/model_eog.pkl')

    def test_saved_model_contract(self):
        self.assertEqual(self.model.n_features_in_, 209)
        self.assertEqual(self.model.classes_.tolist(), [0, 1, 2, 3])
        self.assertEqual(CLASSES, ['notr', 'sag', 'sol', 'kirpma'])
        self.assertEqual(self.model.steps[0][0], 'standardscaler')
        self.assertEqual(self.model.steps[-1][1].hidden_layer_sizes, (96, 48))

    def test_each_recording_accepts_event_window(self):
        for path in sorted((ROOT / 'data/raw').glob('*.csv')):
            with self.subTest(path=path.name):
                signal = np.loadtxt(path)
                self.assertGreater(len(signal), 363)
                vector = hazirla(signal[:363])
                self.assertEqual(vector.shape, (209,))
                self.assertTrue(np.isfinite(vector).all())
                probability = self.model.predict_proba(vector.reshape(1, -1))[0]
                self.assertEqual(probability.shape, (4,))
                self.assertTrue(np.isfinite(probability).all())
                self.assertAlmostEqual(float(probability.sum()), 1.0)

    def test_preprocessing_removes_offset_and_scale(self):
        signal = np.sin(np.linspace(0, 8, 363)) * 200
        np.testing.assert_allclose(hazirla(signal), hazirla(signal * 3 + 8000), rtol=1e-7, atol=1e-7)

    def test_invalid_and_constant_signals(self):
        self.assertIsNone(hazirla([1] * 9))
        self.assertIsNone(hazirla([float('nan')] * 40))
        self.assertIsNone(hazirla([float('inf')] * 40))
        self.assertTrue(np.isfinite(hazirla([1] * 40)).all())

    def test_trainer_matches_supplied_model_architecture(self):
        pipeline = make_model()
        self.assertEqual([type(x).__name__ for _, x in pipeline.steps], ['StandardScaler', 'MLPClassifier'])
        actual = self.model.steps[-1][1].get_params()
        for key, value in pipeline.steps[-1][1].get_params().items():
            self.assertEqual(value, actual[key], key)

    def test_dataset_loader_preserves_labels_and_records_bad_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for folder in ['NOTR', 'SAG', 'SOL', 'GOZ_KIRPMA']:
                path = base / folder / 'participant'
                path.mkdir(parents=True)
                np.savetxt(path / 'event.csv', np.sin(np.linspace(0, 8, 363)))
            bad = base / 'NOTR/participant/bad.csv'
            bad.write_text('not a number', encoding='utf-8')
            X, y, manifest, skipped = load_events(base)
            self.assertEqual(X.shape, (4, 209))
            self.assertEqual(y.tolist(), [0, 1, 2, 3])
            self.assertEqual(len(manifest), 4)
            self.assertEqual(len(skipped), 1)

    def test_reported_confusion_matrix_arithmetic(self):
        cm = np.array([[61, 0, 0, 0], [0, 39, 0, 1], [0, 1, 23, 1], [0, 5, 0, 66]])
        self.assertEqual(cm.sum(), 197)
        self.assertEqual(np.trace(cm), 189)
        self.assertEqual(round(float(np.trace(cm) / cm.sum() * 100), 1), 95.9)

    def _run_gui_without_hardware(self, script, argv, motor_enabled):
        started = []
        class Widget:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        tkinter = types.ModuleType('tkinter')
        tkinter.Tk = lambda *a, **k: Widget()
        tkinter.Label = lambda *a, **k: Widget()
        class NoStartThread:
            def __init__(self, target, **kwargs):
                self.target = target
            def start(self):
                started.append(self.target.__name__)
        env = {'EOG_ENABLE_MOTOR': str(int(motor_enabled)), 'EOG_AUTO_START': '0',
               'EOG_MODEL_PATH': str(ROOT / 'models/model_eog.pkl')}
        with patch.dict(sys.modules, {'tkinter': tkinter}), patch('threading.Thread', NoStartThread), \
             patch.dict(os.environ, env), patch.object(sys, 'argv', argv):
            state = runpy.run_path(str(ROOT / 'src' / script))
        return started, state

    def test_csv_replay_never_connects_motor_even_when_enabled(self):
        started, _ = self._run_gui_without_hardware('motor_control.py',
            ['motor_control.py', str(ROOT / 'data/raw/sag.csv')], True)
        self.assertEqual(started, ['csv_oynat'])

    def test_live_motor_is_disabled_by_default_and_opt_in(self):
        disabled, _ = self._run_gui_without_hardware('motor_control.py', ['motor_control.py'], False)
        enabled, _ = self._run_gui_without_hardware('motor_control.py', ['motor_control.py'], True)
        self.assertEqual(disabled, ['seri_oku'])
        self.assertEqual(enabled, ['seri_oku', 'motor_baglan'])

    def test_eog_runtime_returns_neutral_for_no_event(self):
        started, state = self._run_gui_without_hardware('eog_only.py', ['eog_only.py'], False)
        self.assertEqual(started, ['seri_oku'])
        state['buffer'].extend([2000.0] * 510)
        self.assertEqual(state['tahmin_et'](), 'notr')

if __name__ == '__main__':
    unittest.main()
