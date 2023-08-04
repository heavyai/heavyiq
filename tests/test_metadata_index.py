import unittest

from heavyiq.langchain.index import get_heavydb_index, HeavyDBMetadataIndex


class TestConversationalAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata_index = get_heavydb_index()

    def test_get_heavydb_index(self):
        self.assertIsInstance(self.metadata_index, HeavyDBMetadataIndex)

    def test_simple_search_for_table_names(self):
        table_names = self.metadata_index.simple_search_for_table_names("What table contains data about stock prices?")
        self.assertIsInstance(table_names, list)
        self.assertIn("sp500_2018_2020_minute", table_names)

    def test_ask_about_database(self):
        res = self.metadata_index.ask_about_database("What table contains data about stock prices?")
        self.assertEqual(res["tables"], ["sp500_2018_2020_minute"])


if __name__ == "__main__":
    unittest.main()
