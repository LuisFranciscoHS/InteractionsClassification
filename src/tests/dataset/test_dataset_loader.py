import numpy as np
from unittest import TestCase

import sklearn
from sklearn.datasets.base import Bunch
from dataset.dataset_loader import load_dataset, merge_datasets, merge_features


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
        self.dataset1 = sklearn.utils.Bunch(features=np.array([[1], [0], [1], [0], [1]]),
                                            targets=np.array([True, False, True, False, True]),
                                            interactions=['A', 'B', 'C', 'D', 'E'])

        self.dataset2 = sklearn.utils.Bunch(features=np.array([[1], [0], [1]]),
                                            targets=np.array([True, False, True]),
                                            interactions=['F', 'G', 'H'])

    def test_merge_datasets_no_dataset_raises_error(self):
        with self.assertRaises(ValueError):
            merge_datasets()

    def test_merge_datasets_one_dataset(self):
        self.dataset1 = merge_datasets(self.dataset1)
        self.assertEqual((5, 1), self.dataset1['features'].shape, msg='The length of the features vector should not change.')
        self.assertEqual((5,), self.dataset1['targets'].shape, msg='The length of the targets vector should not change.')
        self.assertEquals(5, len(self.dataset1['interactions']), msg='The number of interactions should not change')

    def test_merge_datasets_merges_multiple(self):
        dataset3 = merge_datasets(self.dataset1, self.dataset2)

        self.assertEqual((8, 1), dataset3['features'].shape, msg='Wrong length of features vector')
        self.assertEqual((8,), dataset3['targets'].shape, msg='Wrong length of target vector')
        self.assertEquals(8, len(dataset3['interactions']), msg='Wrong length of interactions list')

    def test_merge_datasets_wrong_dimmensions(self):
        """Raise error when datasets have different set of features."""
        dataset1 = sklearn.utils.Bunch(features=np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]),
                                       targets=np.array([True, False, True, False, True]),
                                       interactions=['A', 'B', 'C', 'D', 'E'])

        dataset2 = sklearn.utils.Bunch(features=np.array([[1, 0, 1]]),
                                       targets=np.array([True, False, True]),
                                       interactions=['F', 'G', 'H'])

        with self.assertRaises(ValueError, msg="Should not be possible to merge with different number of features"):
            merge_datasets(dataset1, dataset2)


class TestMergeFeatures(TestCase):

    def setUp(self):

        self.v1 = [1, 1, 1, 1]     # One feature, four samples
        self.v2 = [2, 2, 2, 2]     # One feature, four samples
        self.v3 = [3, 3, 3, 3]     # One feature, four samples
        self.features1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])   # 2 samples with four features
        self.v4 = [5, 5]                       # 2 samples with one feature
        self.features3 = np.array([[1, 2, 3]])                    # 1 sample with three features

    def test_merge_features_one_feature(self):
        features = merge_features(self.v1)
        self.assertEqual(np.ndarray, type(features))
        self.assertEqual((4,), features.shape)

    def test_merge_features_two_vectors(self):
        features = merge_features(self.v1, self.v2)
        self.assertEqual(np.ndarray, type(features))
        self.assertEqual((4, 2), features.shape)

    def test_merge_features_multiple(self):
        features = merge_features(self.v1, self.v2, self.v3)
        self.assertEqual((4, 3), features.shape)

    def test_merge_features_wrong_number_of_samples(self):
        with self.assertRaises(ValueError):
            merge_features(self.v1, self.v4)

