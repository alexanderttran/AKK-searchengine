"""
CORE COMPONENT: High-Speed Low-Memory Document Retrieval (Milestone 2)
---------------------------------------------------------------------
TEAM BREAKDOWN (Kenrich, Alexander, Kenny):
This class handles interactive user query execution. It loads our lightweight
mapping assets into RAM at startup and maintains a persistent, open read stream
to our flat file database on disk to minimize query latency.
"""

import json
import math
import re
import time
from pathlib import Path
from parser_utils import get_stem

GLOBAL_INDEX_PATH = Path("global_index.txt")
LEXICON_PATH = Path("lexicon.json")
URL_MAP_PATH = Path("url_map.json")
TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+')

class SearchEngine:
    def __init__(self):
        """Initializes the search engine by verifying required files on disk,
        loading the lexicon configuration map, and inverting the doc_id-to-URL
        dictionary map into RAM for O(1) text lookups. Also establishes a
        persistent binary read handle onto the global index flat file.
        """
        print("Initializing search engine...")

        if not LEXICON_PATH.exists() or not URL_MAP_PATH.exists() or not GLOBAL_INDEX_PATH.exists():
            raise FileNotFoundError(
                "Missing required index assets, make sure you run main.py and merge.py first."
            )

        self.lexicon = self._load_lexicon()
        self.url_map = self._load_url_map()
        self.f_index = open(GLOBAL_INDEX_PATH, 'rb')

        print(f"Search engine initialized, vocabulary size: {len(self.lexicon)} terms")

    def _load_lexicon(self):
        """Loads the JSON lexicon dictionary mapping terms to disk byte offsets."""
        with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_url_map(self):
        """Loads the JSON URL map and inverts it to map string doc IDs to URLs."""
        with open(URL_MAP_PATH, 'r', encoding='utf-8') as f:
            raw_map = json.load(f)
            return {str(doc_id): url for url, doc_id in raw_map.items()}

    def close(self):
        """Safely terminates infrastructure connections by flushing buffers and
        closing the open disk file stream handle to prevent memory leaks.
        """
        if hasattr(self, 'f_index') and self.f_index:
            self.f_index.close()
            print("Global index disk handle closed succesfully")

    def _fetch_postings(self, stem):
        """Performs an O(1) physical disk jump using the lexicon coordinate map.
        Seeks to the absolute byte offset location within the flat-file database,
        streams a single line payload, and decodes the JSON posting data.
        """
        if stem not in self.lexicon:
            print("Stem is not in lexicon")
            return []

        byte_offset = self.lexicon[stem]
        print(f"Token '{stem}' mapped to absolute byte offset: {byte_offset}")
        self.f_index.seek(byte_offset)

        current_line = self.f_index.readline().decode('utf-8').strip()
        token, posting = current_line.split('|', 1)
        postings_list = json.loads(posting)

        print(f"Successfully fetched {len(postings_list)} postings")

        return postings_list

    def _intersect_two_lists(self, list_a, list_b):
        """Performs a linear two-pointer Boolean AND intersection between two sorted postings lists."""
        intersected = []
        a, b = 0, 0

        while a < len(list_a) and b < len(list_b):
            doc_id_a = list_a[a][0]
            doc_id_b = list_b[b][0]

            if doc_id_a == doc_id_b:
                intersected.append(list_a[a])
                a += 1
                b += 1
            elif doc_id_a < doc_id_b:
                a += 1
            else:
                b += 1

        return intersected

    def _intersect_postings(self, query_stems, term_postings):
        """Executes a linear, fast two-pointer intersection (Boolean AND) across
        multiple sorted postings arrays. Sorts the arrays by length (shortest first)
        upfront to minimize comparisons and optimize processing speed.
        """
        if not term_postings:
            return []
        if len(term_postings) == 1:
            return term_postings[0]

        #sort collection of lists by ascending length
        sorted_lists = sorted(term_postings, key=len)

        #baseline result is the shortest list
        result = sorted_lists[0]

        #intersect the baseline against larger lists to find intersections
        for next_list in sorted_lists[1:]:
            result = self._intersect_two_lists(result, next_list)

            #if intersection becomes empty, stop immediately
            if not result:
                break

        return result

    def _calculate_idf_map(self, query_stems, term_postings_dict, total_docs):
        """Precomputes Inverse Document Frequencies (IDF) for each query term."""
        #idf = inverse document frequency
        idf_map = {}
        for stem in query_stems:
            doc_freq = len(term_postings_dict[stem])
            idf_map[stem] = math.log10(total_docs / doc_freq) if doc_freq > 0 else 0
        return idf_map

    def _score_single_document(self, base_posting, query_stems, term_postings_dict, idf_map):
        """Calculates the cumulative log-dampened TF-IDF score for a single document across all query terms."""
        target_doc_id = base_posting[0]
        doc_score = 0.0

        # Iterates through each query term to fetch its stats via binary search,
        # accumulating the weighted TF, IDF, and structural boost into a final score.
        for stem in query_stems:
            stem_postings = term_postings_dict[stem]
            posting = self._binary_search_posting(stem_postings, target_doc_id)

            if posting:
                doc_id, term_freq, positions, importance = posting

                weighted_term_freq = 1 + math.log10(term_freq)
                idf = idf_map[stem]
                boost = 2.5 if importance == 1 else 1.0

                doc_score += weighted_term_freq * boost * idf

        return doc_score

    def _rank_documents(self, intersected_docs, query_stems, term_postings_dict):
        """Calculates log-dampened TF-IDF relevance scores for each matching document.
        Precomputes Inverse Document Frequencies (IDF) per query term, uses binary
        search to look up missing term statistics, applies a 2.5x multiplier boost
        for HTML title/header matches, and sorts the final list descending.
        """
        total_docs = len(self.url_map)
        scored_documents = []

        idf_map = self._calculate_idf_map(query_stems, term_postings_dict, total_docs)

        # Calculates the combined TF-IDF score for each document that survived the
        # intersection filtering, pairing the final score with the document data.
        for base_posting in intersected_docs:
            doc_score = self._score_single_document(base_posting, query_stems, term_postings_dict, idf_map)
            scored_documents.append((doc_score, base_posting))

        #we want highest scores on top
        scored_documents.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in scored_documents]

    def _get_query_stems(self, query_str):
        """Extracts alphanumeric tokens from a query string and reduces them to root stems."""
        raw_tokens = TOKEN_PATTERN.findall(query_str.lower())
        return [get_stem(token) for token in raw_tokens]

    def _fetch_all_postings(self, query_stems):
        """Retrieves postings lists for all stems, caching duplicate terms to prevent redundant I/O."""
        #retrieve postings arrays from disk and store in a dictionary
        #stem : posting
        term_postings_dict = {}
        for stem in query_stems:
            if stem in term_postings_dict:
                continue
            postings = self._fetch_postings(stem)
            if not postings:
                #if even just one term has no matches, total AND intersection is empty
                return None
            term_postings_dict[stem] = postings
        return term_postings_dict

    def _resolve_top_urls(self, ranked_postings, limit=5):
        """Resolves internal document IDs to human-readable URLs for the top N results."""
        #turn top 5 docIDs into real string URLs
        final_results = []
        for match in ranked_postings[:limit]:
            doc_id = match[0]
            url = self.url_map.get(str(doc_id), "URL Not Found")
            final_results.append((url, match))
        return final_results

    def search(self, query_str):
        """Master orchestration pipeline method. Tokenizes raw string text inputs,
        stems them, fetches matching posting records from disk (with duplicate-term
        caching), evaluates the multi-term intersection, ranks findings via TF-IDF,
        and resolves the top 5 internal DocIDs to human-readable string URLs.
        """
        query_stems = self._get_query_stems(query_str)
        if not query_stems:
            return []

        term_postings_dict = self._fetch_all_postings(query_stems)
        if term_postings_dict is None:
            return []

        #find postings that intersect with the queries
        postings_list = list(term_postings_dict.values())
        matched_postings = self._intersect_postings(query_stems, postings_list)

        if not matched_postings:
            return []

        ranked_postings = self._rank_documents(matched_postings, query_stems, term_postings_dict)

        return self._resolve_top_urls(ranked_postings)

    def _binary_search_posting(self, postings_list, target_doc_id):
        """Finds a specific doc_id in a sorted list of postings using O(log M) binary search.
        Returns the entire posting tuple if found, otherwise None."""
        left, right = 0, len(postings_list) - 1

        while left <= right:
            mid = (left + right) // 2
            current_doc_id = postings_list[mid][0]

            if current_doc_id == target_doc_id:
                return postings_list[mid]
            elif current_doc_id < target_doc_id:
                left = mid + 1
            else:
                right = mid - 1

        return None


def main():
    """Operational application entry point loop. Instantiates search structures
    and sequentially loops through the 4 mandatory assignment benchmark queries,
    calculating execution speeds and outputting the top 5 relevant URL paths
    to the command line interface.
    """
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return


    poor_queries = [
        # POOR EFFECTIVENESS: Ubiquitous terms yield a 48.00 ms search but returns irrelevant pages
        "computer science",

        # POOR EFFECTIVENESS: Natural conversational phrasing runs in 165.41 ms but returns irrelevant pages
        "what are the software engineering bachelors graduation requirements for uci",

        # POOR EFFECTIVENESS: Long structural phrase hits 170.31 ms but returns irrelevant pages
        "what are the prerequisite classes for machine learning",

        # POOR EFFECTIVENESS: High-frequency terms execute in a rapid 16.83 ms but returns irrelevant pages
        "software engineering",

        # POOR EFFECTIVENESS: Conversational string hits 135.89 ms but returns irrelevant pages
        "how do I apply for graduation in computer science bachelors uci",

        # POOR EFFECTIVENESS: Specific query runs in 204.19 ms but returns irrelevant pages
        "what is the room number of professor lopes office",

        # POOR EFFECTIVENESS: High-frequency words run in a fast 95.81 ms but returns irrelevant pages
        "research project and systems",

        # POOR EFFECTIVENESS: Extremely fast 3.28 ms lookup but returns irrelevant pages
        "phd graduate advisor",

        # POOR EFFECTIVENESS: Runs efficiently in 25.68 ms but returns irrelevant pages
        "science data center",

        # POOR EFFECTIVENESS: Completes in a fast 26.20 ms but returns irrelevant pages
        "information technology",

        # POOR EFFECTIVENESS: URL tokens run in 78.54 ms but return a wall of near-duplicate historical DokuWiki parameters (?rev=).
        "https://ics.uci.edu",

        # POOR EFFECTIVENESS: URL tokens run in 13.91 ms but return irrelevant pages
        "algorithmic design",

        # ZERO RESULTS: Completes in under 1.00 ms but strict AND logic crashes on "cv" because no page contains all three terms.
        "raymond klefstad cv",

        # ZERO RESULTS: Long conversational fallback runs in 162.32 ms but drops to empty because of strict multi-pointer matching.
        "how do I apply for transfer from community college to uci software engineering major bachelors undergraduate fall 2027 requirements",
    ]

    good_queries = [
        # GOOD PERFORMANCE: Highly efficient 5.93 ms execution owing to strict two-pointer subset pruning on a small postings intersection.
        "lopes research",

        # GOOD PERFORMANCE: Sub-millisecond 0.09 ms search utilizing a single unique token lookup to directly pull foundational homepages.
        "mondego",

        # GOOD PERFORMANCE: Completes in 20.87 ms, demonstrating superb list filtering and ranking by mapping concise office portals.
        "ics student affairs",

        # GOOD PERFORMANCE: Runs efficiently in 12.77 ms, successfully isolating focused curricular tracks via short keyword groupings.
        "informatics core course",

        # GOOD PERFORMANCE: Resolves in 15.14 ms as prominent institutional heading tags match the target terms to elevate authoritative anchors.
        "donald bren school",

        # GOOD PERFORMANCE: Executes in 26.51 ms, matching specialized directory layouts by leveraging structural header importance tags.
        "computer science lecture schedule",

        # GOOD PERFORMANCE: Fast 2.64 ms query latency achieved via shortest-first sorting optimizations on a small bounding postings set.
        "cybersecurity lab",

        # GOOD PERFORMANCE: High-efficiency 2.49 ms runtime due to a highly constrained comparison window inside the main evaluation block.
        "medical informatics",

        # GOOD PERFORMANCE: Solid 37.70 ms execution speed despite high-frequency terms, thanks to optimized linear array traversal loops.
        "computing systems",

        # GOOD PERFORMANCE: Finishes in 13.94 ms, validating excellent suffix reduction and index matching via grammatical root stemming.
        "algorithmic design",

        # GOOD PERFORMANCE: Fast 17.12 ms retrieval processing that accurately identifies specialized academic portal nodes using rare term weights.
        "machine learning",
    ]

    print("\n========================================================")
    print("      UCI ICS SEARCH ENGINE - M2 BENCHMARK RUNNER       ")
    print("========================================================\n")

    for query in test_queries: #replace with good or poor queries
        print(f"Executing Query: '{query}'")

        start_time = time.time()
        results = engine.search(query)
        duration = (time.time() - start_time) * 1000  # Convert to ms

        print(f"Search completed in {duration:.2f} ms.")
        print("-" * 70)

        if not results:
            print("  No documents matched all search terms.")
        else:
            for idx, (url, match_data) in enumerate(results, 1):
                doc_id, tf, positions, importance = match_data
                print(f" {idx}. {url}")
        print("========================================================\n")

    print("Benchmarking complete. Shutting down engine...")
    engine.close()


if __name__ == "__main__":
    main()