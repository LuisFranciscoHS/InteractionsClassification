import os
import sys
from unittest import TestCase

import numpy as np
import sklearn

from config import read_config
from dataset import reactome


class Test_reactome(TestCase):

    def setUp(self):
        print(os.getcwd())
        config = read_config('../../../')
        self.result = reactome.load_dataset(config)

    def test_load_dataset_returns_bunch(self):
        self.assertTrue(type(self.result) == sklearn.utils.Bunch)

    def test_load_dataset_has_target_key(self):
        self.assertIn('targets', self.result.keys())

    def test_load_dataset_has_features_key(self):
        self.assertIn('features', self.result.keys())

    def test_load_dataset_has_accessions_key(self):
        self.assertIn('interactions', self.result.keys())

    def test_target_type(self):
        self.assertEqual(np.ndarray, type(self.result['targets']), msg="'target' value has wrong type")

    def test_features_type(self):
        self.assertEqual(np.ndarray, type(self.result['features']), msg="'features' value has wrong type")

    def test_interactions_type(self):
        self.assertEqual(list, type(self.result['interactions']), msg="'interactions' value has wrong type")

    def test_dataset_shape_consistency(self):
        self.assertEqual(len(self.result['interactions']), self.result['features'].shape[0],
                         msg="There should be a pair if interacting proteins for each sample in features.")
        self.assertEqual(self.result['features'].shape[0], self.result['targets'].shape[0],
                         msg="There should be a target value for each sample in the features.")

    def test_load_dataset_target_all_true(self):
        unique, counts = np.unique(self.result['targets'], return_counts=True)
        self.assertEqual(len(self.result['targets']), counts[True],
                         msg='The target vector of the reactome dataset should all be True')
