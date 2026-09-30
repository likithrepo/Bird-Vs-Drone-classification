import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from radar import CNN, preprocess, metrics, threshold_for, generate, dataset

class RadarTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        model = CNN(); x = np.random.default_rng(3).random((2, 48, 48)).astype('float32'); y = np.array([0, 1])
        grads = model.gradients(x, y)
        def loss():
            p = model.forward(x)[0].astype('float64')
            return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
        for key, index in [('kernel', (1, 2, 2)), ('weight', (2,)), ('bias', (3,)), ('output', (0,))]:
            old = model.params[key][index]; eps = .002
            model.params[key][index] = old+eps; plus = loss()
            model.params[key][index] = old-eps; minus = loss()
            model.params[key][index] = old
            self.assertAlmostEqual(float(grads[key][index]), (plus-minus)/(2*eps), delta=.002)

    def test_constant_image_is_finite(self):
        x = preprocess(Image.new('L', (20, 20), 0))
        self.assertEqual(x.shape, (48, 48)); self.assertTrue(np.isfinite(x).all())

    def test_threshold_controls_false_alarms(self):
        y = np.array([0, 0, 1, 1]); p = np.array([.1, .6, .7, .9])
        t = threshold_for(y, p, 1)
        self.assertEqual(metrics(y, p, t)['bird_false_alarm_rate'], 0)
        self.assertEqual(metrics(y, p, t)['drone_recall'], 1)

    def test_roundtrip_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); generate(root/'data', 10)
            sets = dataset(root/'data'); self.assertEqual(len(sets['train'][1]), 20)
            model = CNN(); model.save(root/'model.npz', .7, True)
            loaded = CNN.load(root/'model.npz')
            np.testing.assert_allclose(model.forward(sets['test'][0])[0], loaded.forward(sets['test'][0])[0])
            source = root/'data/train/bird/0000.png'; dest = root/'data/test/bird/copied.png'
            dest.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate'): dataset(root/'data')

if __name__ == '__main__': unittest.main()
