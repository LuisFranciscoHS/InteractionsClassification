import os
from unittest import TestCase

from dataset.download import download_if_not_exists


class Test_download(TestCase):
    def test_download_if_not_exists_binary(self):
        filename = 'test_download_if_not_exists_binary_PathwayMatcher.jar'
        url = 'https://github.com/PathwayAnalysisPlatform/PathwayMatcher/releases/download/1.9.1/PathwayMatcher.jar'
        self.assertFalse(os.path.exists(filename))
        download_if_not_exists('./', filename, url, 'PathwayMatcher')
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_download_if_not_exists_empty_path(self):
        filename = 'test_download_if_not_exists_empty_path_README.md'
        url = 'https://github.com/PathwayAnalysisPlatform/PathwayMatcher/blob/master/README.md'
        self.assertFalse(os.path.exists(filename))
        download_if_not_exists('', filename, url, 'README')
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_download_if_not_exists_text(self):
        filename = 'test_download_if_not_exists_text_README.md'
        url = 'https://github.com/PathwayAnalysisPlatform/PathwayMatcher/blob/master/README.md'
        self.assertFalse(os.path.exists(filename))
        download_if_not_exists('./', filename, url, 'README')
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)
