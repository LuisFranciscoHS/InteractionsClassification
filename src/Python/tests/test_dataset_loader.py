import unittest

from src.Python.datasets.dataset_loader import read_swissprot_proteins


class Test_dataset_loader(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_create_targets_returns_dataframe(self):
        self.fail("Test not implemented")

    def test_get_create_targets_returns_three_column_dataframe(self):
        self.fail("Test not implemented")

    def test_get_create_targets_returns_number_of_dataframe_rows_correct(self):
        self.fail("Test not implemented")

    def test_get_create_targets_protein_columns_are_strings(self):
        self.fail("Test not implemented")

    def test_get_create_targets_label_is_boolean(self):
        self.fail("Test not implemented")

    def test_get_create_targets_interactions_in_reactome_are_functional(self):
        self.fail("Test not implemented")

    def test_get_create_targets_interactions_not_in_reactome_and_low_string_score_are_not_functional(self):
        self.fail("Test not implemented")

    def test_get_create_targets_interactions_not_in_reactome_and_not_in_string_are_not_functional(self):
        self.fail("Test not implemented")

    def test_get_create_targets_interactions_not_in_reactome_and_high_string_score_are_ignored(self):
        self.fail("Test not implemented")

    def test_read_swissprot_proteins_returns_list(self):
        self.fail("Test not implemented")

    def test_read_swissprot_downloads_file_if_not_exists(self):
        # Verify that the file does not exist

        # Execute the method
        self.assertEquals(list, type(read_swissprot_proteins()), msg="The return type was not correct.")

        # Verify that the file was created

        # Check file has expected format
        # Check file has expected number of proteins

        # Remove file to end the test
        self.fail("Test not implemented")

    def test_read_swissprot_list_has_correct_types(self):
        self.fail("Test not implemented")

    def test_read_swissprot_list_has_correct_proteins(self):
        self.fail("Test not implemented")

# Check only