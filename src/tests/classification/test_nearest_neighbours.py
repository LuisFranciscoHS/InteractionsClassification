import importlib
import unittest

from classification import nearest_neighbours
from dataset import dataset_loader


class Test_nearest_neighbours(unittest.TestCase):

    def setUp(self):
        """
        Create dataset to test the Nearest Neighbours classification
        """
        importlib.reload(dataset_loader)
        importlib.reload(nearest_neighbours)
        self.dataset = dataset_loader.load_dataset()

    def test_returns_float_score(self):
        score = nearest_neighbours.classify(self.dataset)
        self.assertTrue(type(score) == type(0.0))

if __name__ == '__main__':
    unittest.main()
