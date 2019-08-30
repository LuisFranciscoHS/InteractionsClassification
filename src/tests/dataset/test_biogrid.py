import os
from unittest import TestCase

from config_loader import read_config
from dataset import biogrid
from dataset.biogrid import create_ggi_file
from dictionaries import create_ppis_dictionary


class Test_biogrid(TestCase):
    def test_get_interactions_file_exists_reads_dictionary(self):
        # Create file with ppis
        ppis = [('Q15051', 'Q5U5U6'), ('Q15051', 'V9HWP2'), ('A0A024R823', 'Q92616'), ('A0A024R823', 'Q6FIF0'),
                ('Q59H94', 'Q9UKB1'), ('Q59H94', 'Q9UKB1'), ('P33992', 'Q7Z6C1'), ('B1AHB0', 'Q12906')]
        filename_ppis = 'test_get_interactions_file_exists_reads_dictionary.txt'
        with open(filename_ppis, 'w') as file_ppis:
            for ppi in ppis:
                file_ppis.write(f"{ppi[0]}\t{ppi[1]}\n")
        # Check dictionary is right
        result = biogrid.get_interactions('./', '', '', '', filename_ppis, '', 1)
        self.assertEqual(dict, type(result))
        self.assertEqual(5, len(result.keys()))

        os.remove(filename_ppis)

    def test_get_interactions_file_not_exists_mapping_and_ggis_exist(self):
        """File for the ppis does not exist, it is created. Files ggis and entrez to uniprot exist."""
        filename_ppis = TestCase.id(self) + '_ppis.txt'
        filename_ggis = TestCase.id(self) + '_ggis.txt'
        filename_entrez_to_uniprot = TestCase.id(self) + '_entrez_to_uniprot.txt'

        # Create entrez_to_uniprot file
        with open(filename_entrez_to_uniprot, 'w') as file_entrez_to_uniprot:
            file_entrez_to_uniprot.write('6416\tP45985\n84665\tA0A087WX60\n84665\tQ86TC9\n90\tD3DPA4\n90\tQ04771\n')
            file_entrez_to_uniprot.write('2624\tP23769\n6118\tB4DUL2\n6118\tP15927\n')

        # Create ggis file
        with open(filename_ggis, 'w') as file_ggis:
            file_ggis.write('BioGRID Interaction ID\tEntrez Gene Interactor A\tEntrez Gene Interactor B\tOther1\n')
            file_ggis.write('0\t6416\t84665\t0\n' +
                            '1\t90\t2624\t1\n' +
                            '2\t84665\t6118\t2\n')

        # Assert preconditions
        self.assertFalse(os.path.exists(filename_ppis), msg='As precondition ppis file should not exist')
        self.assertTrue(os.path.exists(filename_ggis), msg='As precondition ggis file should exist')
        self.assertTrue(os.path.exists(filename_entrez_to_uniprot),
                        msg='As precondition entrez_to_uniprot file should exist')
        # Execute target method
        result = biogrid.get_interactions('', '', '', filename_ggis, filename_ppis, filename_entrez_to_uniprot, 1)

        # Check if dictionary is right
        try:
            self.assertEqual(dict, type(result))
            self.assertEqual(6, len(result.keys()))
            for key in ['A0A087WX60', 'B4DUL2', 'D3DPA4', 'P15927', 'P23769', 'P45985']:
                self.assertIn(key, result.keys(), msg=f"Missing key {key}")
            self.assertIn('Q86TC9', result['B4DUL2'])
            self.assertIn('Q04771', result['P23769'])
            self.assertIn('Q86TC9', result['P45985'])
        finally:
            os.remove(filename_ggis)
            os.remove(filename_ppis)
            os.remove(filename_entrez_to_uniprot)

    def test_create_ggi_file_file_exists_do_nothing(self):
        # Create mock file
        filename_ggis = TestCase.id(self) + '_ggis.txt'
        file_ggis = open(filename_ggis, 'w')
        file_ggis.close()

        self.assertTrue(os.path.exists(filename_ggis), msg='ggis file should exist as precondition')
        create_ggi_file('', '', '', filename_ggis)
        self.assertTrue(os.path.exists(filename_ggis), msg='ggis file should exist after the creation method')
        os.remove(filename_ggis)

    def test_create_ggi_file_file_not_exists_and_create_file(self):
        config = read_config('../../../config.txt')
        self.assertFalse(os.path.exists(config['BIOGRID_GGIS']), msg='ggis file should not exist as precondition')
        create_ggi_file(config['URL_BIOGRID_ALL'], '', config['BIOGRID_ALL'], config['BIOGRID_GGIS'])
        self.assertTrue(os.path.exists(config['BIOGRID_GGIS']), msg='ggis file should exist after the creation method')
        os.remove(config['BIOGRID_GGIS'])
        self.assertFalse(os.path.exists(config['BIOGRID_ALL']), msg="Zip file should be deleted")

    def test_create_ppis_dictionary(self):

        ggis = {'6416': {'84665'}, '90': {'2624'}, '84665': {'6118'}}
        entrez_to_uniprot = {'6416': {'P45985'},
                             '84665': {'A0A087WX60', 'Q86TC9'},
                             '90': {'D3DPA4', 'Q04771'},
                             '2624': {'P23769'},
                             '6118': {'B4DUL2', 'P15927'}}
        ppis = create_ppis_dictionary(ggis, entrez_to_uniprot)

        self.assertEqual(dict, type(ppis))
        self.assertEqual(6, len(ppis.keys()))
        for key in ['A0A087WX60', 'B4DUL2', 'D3DPA4', 'P15927', 'P23769', 'P45985']:
            self.assertIn(key, ppis.keys(), msg=f"Missing key {key}")
        self.assertIn('Q86TC9', ppis['B4DUL2'])
        self.assertIn('Q04771', ppis['P23769'])
        self.assertIn('Q86TC9', ppis['P45985'])


