import os
from unittest import TestCase

from config import append_relative_path, read_config


class Test_config(TestCase):

    def setUp(self):
        self.config = {'PATH_1': 'data/', 'PATH_2': 'tools/', 'VAR': 'value'}

    def test_append_relative_path(self):
        prefix = "../prefix/"
        paths = ['PATH_1', 'PATH_2']
        append_relative_path(self.config, prefix, paths)
        self.assertEqual('value', self.config['VAR'], msg='The variable was modified by mistake.')
        self.assertEqual('../prefix/data/', self.config['PATH_1'], msg='Did not add the prefix to path 1')
        self.assertEqual('../prefix/tools/', self.config['PATH_2'], msg='Did not add the prefix to path 2')


    def test_read_config(self):
        path_config = ''
        file_config = open(path_config + 'config.txt', 'w')
        for k, v in self.config.items():
            file_config.write(f"{k}\t{v}\n")
        file_config.close()

        config = read_config(path_config)

        self.assertIn('PATH_1', config.keys())
        self.assertIn('PATH_2', config.keys())
        self.assertIn('VAR', config.keys())
        self.assertEqual('data/', config['PATH_1'])
        self.assertEqual('tools/', config['PATH_2'])
        self.assertEqual('value', config['VAR'])

        os.remove(path_config + 'config.txt')

    def test_read_config_appending_path(self):
        path_config = '../'
        file_config = open(path_config + 'config.txt', 'w')
        for k, v in self.config.items():
            file_config.write(f"{k}\t{v}\n")
        file_config.close()

        config = read_config(path_config)

        self.assertIn('PATH_1', config.keys())
        self.assertIn('PATH_2', config.keys())
        self.assertIn('VAR', config.keys())
        self.assertEqual(path_config + 'data/', config['PATH_1'])
        self.assertEqual(path_config + 'tools/', config['PATH_2'])
        self.assertEqual('value', config['VAR'])

        os.remove(path_config + 'config.txt')
