import numpy as np
from unittest import TestCase

import sklearn
from sklearn.datasets.base import Bunch
from dataset.dataset_loader import load_dataset, merge_datasets


class TestDatasetLoader(TestCase):

    def setUp(self):
        self.result = load_dataset('../../../')

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


class TestMergeDatasets(TestCase):

    def setUp(self):
        self.dataset1 = sklearn.utils.Bunch(features=np.array([1, 0, 1, 0, 1]),
                                            targets=np.array([True, False, True, False, True]),
                                            interactions=['A', 'B', 'C', 'D', 'E'])

        self.dataset2 = sklearn.utils.Bunch(features=np.array([1, 0, 1]),
                                            targets=np.array([True, False, True]),
                                            interactions=['F', 'G', 'H'])

    def test_merge_datasets_no_dataset_raises_error(self):
        with self.assertRaises(ValueError):
            merge_datasets()

    def test_merge_datasets_one_dataset(self):
        self.dataset1 = merge_datasets(self.dataset1)
        self.assertEqual((5,), self.dataset1['features'].shape, msg='The length of the features vector should not change.')
        self.assertEqual((5,), self.dataset1['targets'].shape, msg='The length of the targets vector should not change.')
        self.assertEquals(5, len(self.dataset1['interactions']), msg='The number of interactions should not change')

    def test_merge_datasets_merges_multiple(self):
        dataset3 = merge_datasets(self.dataset1, self.dataset2)

        self.assertEqual((8,), dataset3['features'].shape, msg='Wrong length of features vector')
        self.assertEqual((8,), dataset3['targets'].shape, msg='Wrong length of target vector')
        self.assertEquals(8, len(dataset3['interactions']), msg='Wrong length of interactions list')


class TestMergeFeatures(TestCase):

    def test_merge_features_one_feature(self):
        self.fail()

    def test_merge_features_multiple(self):
        self.fail()
