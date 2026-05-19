import unittest
from unittest.mock import patch, mock_open, call
import math

# Import your SearchEngine class
from search import SearchEngine


class TestSearchEngine(unittest.TestCase):

    @patch('search.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def setUp(self, mock_json_load, mock_file_open, mock_path_exists):
        """Mocks out all disk I/O dependency gates to instantiate a clean,
        isolated SearchEngine instance in RAM before every test."""

        mock_path_exists.return_value = True

        # Mock json.load returns: Lexicon first, then URL Map (URL: DocID format)
        mock_json_load.side_effect = [
            {"machin": 0, "learn": 100, "appl": 200},
            {"https://uci.edu": 0, "https://ics.uci.edu": 5}
        ]

        self.engine = SearchEngine()

    def tearDown(self):
        """Cleans up memory references after each test execution completes."""
        if hasattr(self, 'engine') and self.engine.f_index:
            self.engine.f_index.close()

    # -------------------------------------------------------------------------
    # UNIT TESTS: _binary_search_posting
    # -------------------------------------------------------------------------

    def test_binary_search_found_exact(self):
        """Verifies O(log M) binary search correctly identifies a target DocID."""
        postings = [
            [0, 5, [1, 2, 3], 0],
            [5, 12, [10, 14], 1],
            [12, 2, [45], 0]
        ]

        result = self.engine._binary_search_posting(postings, 5)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 5)
        self.assertEqual(result[1], 12)
        self.assertEqual(result[3], 1)

    def test_binary_search_not_found_returns_none(self):
        """Verifies the fallback safety mechanism returns None instead of crashing."""
        postings = [
            [0, 5, [1], 0],
            [12, 2, [45], 0]
        ]
        result = self.engine._binary_search_posting(postings, 5)
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # UNIT TESTS: _intersect_postings
    # -------------------------------------------------------------------------

    def test_intersect_postings_empty_and_single(self):
        """Ensures empty and single arrays safely bypass processing loops."""
        self.assertEqual(self.engine._intersect_postings(["test"], []), [])

        single_postings = [[[5, 2, [1], 0]]]
        self.assertEqual(
            self.engine._intersect_postings(["test"], single_postings),
            single_postings[0]
        )

    def test_intersect_postings_match_two_pointer(self):
        """Tests the mathematical correctness of the linear intersector."""
        list_a = [[0, 1, [1], 0], [5, 4, [2], 1], [10, 2, [3], 0]]
        list_b = [[2, 1, [1], 0], [5, 9, [1], 0], [12, 1, [2], 0]]

        result = self.engine._intersect_postings(["a", "b"], [list_a, list_b])

        # Only DocID 5 matches both lists. Must retain List A's structural data.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 5)
        self.assertEqual(result[0][1], 4)

        # -------------------------------------------------------------------------

    # UNIT TESTS: _rank_documents
    # -------------------------------------------------------------------------

    def test_rank_documents_tf_idf_order(self):
        """Validates that mathematical scoring weights correctly calculate local tf,
        global idf, and HTML header tags to return data in descending order."""

        # INCREASE N to 10 so the IDF does not mathematically drop to 0
        self.engine.url_map = {str(i): f"url_{i}" for i in range(10)}

        query_stems = ["machin"]
        postings_term = [
            [0, 1, [1], 0],  # Normal text, low tf
            [1, 5, [1], 1]  # Heading match, high tf -> Must rank first
        ]
        term_postings_dict = {"machin": postings_term}
        intersected_docs = [[0, 1, [1], 0], [1, 5, [1], 1]]

        ranked_results = self.engine._rank_documents(intersected_docs, query_stems, term_postings_dict)

        # Document 1 must scale past Document 0
        self.assertEqual(len(ranked_results), 2)
        self.assertEqual(ranked_results[0][0], 1)
        self.assertEqual(ranked_results[1][0], 0)

    # -------------------------------------------------------------------------
    # UNIT TESTS: search Master Pipeline & Optimizations
    # -------------------------------------------------------------------------

    @patch('search.get_stem')
    def test_search_duplicate_stem_optimization(self, mock_stem):
        """Proves that redundant query words do not trigger multiple disk reads."""

        # Simulate a query where both words stem to the exact same root
        mock_stem.side_effect = ["appl", "appl"]

        # Intercept the hardware fetch utility to track how many times it gets called
        with patch.object(self.engine, '_fetch_postings') as mock_fetch:
            mock_fetch.return_value = [[0, 1, [1], 0]]

            # Execute search with redundant terms
            self.engine.search("apple apples")

            # The assertion proves the disk was only touched ONCE
            mock_fetch.assert_called_once_with("appl")

    @patch('search.get_stem')
    def test_search_pipeline_no_hits(self, mock_stem):
        """Verifies pipeline gracefully shortcuts to empty on unregistered terms."""
        mock_stem.return_value = "unregistered"

        with patch.object(self.engine, '_fetch_postings', return_value=[]):
            results = self.engine.search("fakequery")
            self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()