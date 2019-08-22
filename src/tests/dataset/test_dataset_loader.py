import numpy as np
from unittest import TestCase
from sklearn.datasets.base import Bunch
from dataset.dataset_loader import load_dataset


class Test_dataset_loader(TestCase):

    def setUp(self):
        self.result = load_dataset()

    def test_load_dataset_returns_bunch(self):
        self.assertTrue(type(self.result) == Bunch)

    def test_load_dataset_has_target_key(self):
        self.assertIn('target', self.result.keys())

    def test_load_dataset_has_features_key(self):
        self.assertIn('features', self.result.keys())

    def test_load_dataset_has_accessions_key(self):
        self.assertIn('interactions', self.result.keys())

    def test_target_type(self):
        self.assertEqual(np.ndarray, type(self.result['target']), msg="'target' value has wrong type")

    def test_features_type(self):
        self.assertEqual(np.ndarray, type(self.result['features']), msg="'features' value has wrong type")

    def test_interactions_type(self):
        self.assertEqual(dict, type(self.result['interactions']), msg="'interactions' value has wrong type")

    def test_dataset_shape_consistency(self):
        self.assertEqual(len(self.result['interactions'].keys()), self.result['features'].shape[0],
                         msg="There should be a pair if interacting proteins for each sample in features.")
        self.assertEqual(self.result['features'].shape[0], self.result['target'].shape[0],
                         msg="There should be a target value for each sample in the features.")
