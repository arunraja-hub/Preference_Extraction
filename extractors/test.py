import unittest
import tf_extractor

class ExtractorTestCase(unittest.TestCase):
    def test_cnn_from_obs(self):
        model = tf_extractor.cnn_from_obs(None, 16, 64, 5, 2, 64, 8, 3, .2, .1)
        self.assertEqual(model.layers[0].filters, 16)
        for i in range(4):
            self.assertGreater(model.layers[i+1].filters, model.layers[i].filters)
        self.assertEqual(model.layers[4].filters, 64)

        self.assertEqual(model.layers[0].strides[0], 1)
        self.assertEqual(model.layers[1].strides[0], 2)
        self.assertEqual(model.layers[2].strides[0], 1)
        self.assertEqual(model.layers[3].strides[0], 2)
        self.assertEqual(model.layers[4].strides[0], 1)

        self.assertEqual(model.layers[6].units, 64)
        for i in range(6, 6+3*2, 2):
            self.assertLess(model.layers[i+2].units, model.layers[i].units)
        self.assertEqual(model.layers[6+3*2-2].units, 8)
        self.assertEqual(model.layers[6 + 3 * 2].units, 1)

    def test_cnn_from_obs_stride(self):
        model = tf_extractor.cnn_from_obs(None, 16, 32, 5, 1, 64, 8, 4, .2, .1)
        self.assertEqual(model.layers[0].filters, 16)
        for i in range(4):
            self.assertGreater(model.layers[i+1].filters, model.layers[i].filters)
        self.assertEqual(model.layers[4].filters, 32)

        self.assertEqual(model.layers[0].strides[0], 2)
        self.assertEqual(model.layers[1].strides[0], 2)
        self.assertEqual(model.layers[2].strides[0], 2)
        self.assertEqual(model.layers[3].strides[0], 2)
        self.assertEqual(model.layers[4].strides[0], 2)

        self.assertEqual(model.layers[6].units, 64)
        for i in range(6, 6+4*2, 2):
            self.assertLess(model.layers[i+2].units, model.layers[i].units)
        self.assertEqual(model.layers[6+4*2-2].units, 8)
        self.assertEqual(model.layers[6 + 4 * 2].units, 1)

if __name__ == '__main__':
    unittest.main()
