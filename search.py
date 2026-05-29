"""
CORE COMPONENT: High-Speed Low-Memory Document Retrieval (Milestone 2)
---------------------------------------------------------------------
TEAM BREAKDOWN (Kenrich, Alexander, Kenny):
This class handles interactive user query execution. It loads our lightweight
mapping assets into RAM at startup and maintains a persistent, open read stream
to our flat file database on disk to minimize query latency.

MILESTONE 3 RANKING IMPROVEMENTS:
Three heuristics from Lectures 21-23 are now implemented:

  1. HIGH-IDF INDEX ELIMINATION (Lec 22, p.38-40)
     Before running the AND intersection, any query term whose IDF falls below
     IDF_THRESHOLD is discarded. Stop-words like "what", "are", "the" have very
     low IDF and almost never change final rankings — dropping them avoids the
     common problem where a ubiquitous word crushes the AND intersection to zero.
     A minimum of 1 term is always kept so single-word queries still work.

  2. SOFT CONJUNCTION FALLBACK (Lec 22, p.41-46)
     The slides recommend relaxing strict AND when too few results surface.
     After strict N-of-N AND, the engine iteratively drops the rarest term
     (lowest IDF) and widens to N-1-of-N, then N-2-of-N, etc., until at
     least MIN_RESULTS docs are found. Documents matching more terms are
     scored higher, so the top results naturally reflect the best matches.

  3. QUERY TERM PROXIMITY SCORING (Lec 23, p.58-63)
     Position lists are already stored in every posting. We now use them:
     for each candidate document, the minimum spanning window (smallest
     contiguous word range containing all query terms) is computed. A
     proximity bonus of PROXIMITY_WEIGHT / (1 + min_window) is added to the
     TF-IDF score — documents where query terms appear close together rank higher.
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

# --- Heuristic Tuning Constants ---
# Heuristic 1: Terms with IDF below this threshold are considered stop-words
# and dropped from the AND intersection before fetching postings.
# IDF = log10(N / df). At N=55,000 docs, IDF < 1.0 means df > 5,500 (~10% of corpus).
IDF_THRESHOLD = 1.0

# Heuristic 2: Minimum results before the soft-conjunction fallback kicks in.
MIN_RESULTS = 5

# Heuristic 3: Weight applied to the proximity bonus (1 / (1 + min_window)).
# Set to 0.0 to disable proximity scoring entirely.
PROXIMITY_WEIGHT = 3.0


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
        self.total_docs = len(self.url_map)
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

    def _calculate_idf_map(self, query_stems, term_postings_dict):
        """Precomputes Inverse Document Frequencies (IDF) for each query term."""
        #idf = inverse document frequency
        idf_map = {}
        for stem in query_stems:
            doc_freq = len(term_postings_dict[stem])
            idf_map[stem] = math.log10(self.total_docs / doc_freq) if doc_freq > 0 else 0
        return idf_map

    # -------------------------------------------------------------------------
    # HEURISTIC 3: Query Term Proximity Scoring (Lecture 23, p. 58-63)
    # -------------------------------------------------------------------------
    def _compute_proximity_bonus(self, target_doc_id, query_stems, term_postings_dict):
        """Computes a proximity bonus for a document based on the minimum spanning
        window (w) across all query terms' position lists.

        The slides define w as the smallest contiguous word-position range
        that contains at least one occurrence of every query term. A smaller w
        means the terms cluster together, which is a strong relevance signal.

        Bonus formula:  PROXIMITY_WEIGHT / (1 + w)
        - When terms are adjacent (w=1), bonus approaches PROXIMITY_WEIGHT.
        - When terms are spread far apart (large w), bonus approaches 0.

        If any query term is missing from the document, returns 0 (no bonus).
        """
        if PROXIMITY_WEIGHT == 0.0 or len(query_stems) <= 1:
            return 0.0

        # Collect the positions list for each query term in this document.
        positions_per_term = []
        for stem in query_stems:
            posting = self._binary_search_posting(term_postings_dict[stem], target_doc_id)
            if posting is None:
                # Term is absent from this document — proximity is undefined.
                return 0.0
            _, _, positions, _ = posting
            if not positions:
                return 0.0
            positions_per_term.append(sorted(positions))

        # Sliding-window minimum-span algorithm:
        # Use one pointer per term, always advance the pointer pointing to the
        # globally smallest position. Track the current span [min_pos, max_pos].
        import heapq
        # Heap entries: (position, term_index, list_index)
        heap = []
        for term_idx, pos_list in enumerate(positions_per_term):
            heapq.heappush(heap, (pos_list[0], term_idx, 0))

        # Track the current max across all term-front pointers
        current_max = max(pos_list[0] for pos_list in positions_per_term)
        min_window = float('inf')

        while True:
            current_min, term_idx, list_idx = heapq.heappop(heap)
            window = current_max - current_min
            if window < min_window:
                min_window = window

            # Advance this term's pointer to its next position
            next_list_idx = list_idx + 1
            if next_list_idx >= len(positions_per_term[term_idx]):
                # This term has no more positions — we can't improve further
                break

            next_pos = positions_per_term[term_idx][next_list_idx]
            heapq.heappush(heap, (next_pos, term_idx, next_list_idx))
            if next_pos > current_max:
                current_max = next_pos

        return PROXIMITY_WEIGHT / (1.0 + min_window)

    def _score_single_document(self, base_posting, query_stems, term_postings_dict, idf_map):
        """Calculates the cumulative log-dampened TF-IDF score for a single document
        across all query terms, plus the query-term proximity bonus.
        """
        target_doc_id = base_posting[0]
        doc_score = 0.0

        # Iterates through each query term to fetch its stats via binary search,
        # accumulating the weighted TF, IDF, and structural boost into a final score.
        for stem in query_stems:
            if stem not in term_postings_dict:
                continue
            stem_postings = term_postings_dict[stem]
            posting = self._binary_search_posting(stem_postings, target_doc_id)

            if posting:
                doc_id, term_freq, positions, importance = posting

                weighted_term_freq = 1 + math.log10(term_freq)
                idf = idf_map.get(stem, 0)
                boost = 2.5 if importance == 1 else 1.0

                doc_score += weighted_term_freq * boost * idf

        # HEURISTIC 3: Add proximity bonus on top of TF-IDF score.
        doc_score += self._compute_proximity_bonus(target_doc_id, query_stems, term_postings_dict)

        return doc_score

    def _rank_documents(self, intersected_docs, query_stems, term_postings_dict):
        """Calculates log-dampened TF-IDF relevance scores for each matching document.
        Precomputes Inverse Document Frequencies (IDF) per query term, uses binary
        search to look up missing term statistics, applies a 2.5x multiplier boost
        for HTML title/header matches, adds proximity bonus, and sorts descending.
        """
        scored_documents = []

        idf_map = self._calculate_idf_map(query_stems, term_postings_dict)

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

    # -------------------------------------------------------------------------
    # HEURISTIC 1: High-IDF Index Elimination (Lecture 22, p. 38-40)
    # -------------------------------------------------------------------------
    def _filter_low_idf_stems(self, query_stems):
        """Drops query terms whose document frequency is so high that their IDF
        falls below IDF_THRESHOLD — these are effectively stop-words that appear
        in too many documents to meaningfully discriminate between results.

        The lecture example: for "catcher in the rye", drop "in" and "the"
        because they contribute almost nothing to ranking while ballooning the
        postings lists that must be intersected.

        Always retains at least one stem (the highest-IDF one) so that single-word
        queries and short queries don't collapse to an empty term set.

        Also drops any stem that doesn't exist in the lexicon at all, since those
        would produce zero results regardless.
        """
        if not query_stems:
            return query_stems

        # Score each stem by its approximate IDF (using its postings length in lexicon).
        # Terms not in the lexicon get IDF=0 and are always dropped.
        stem_idfs = []
        for stem in query_stems:
            if stem in self.lexicon:
                # We need actual postings length — fetch it quickly.
                postings = self._fetch_postings(stem)
                df = len(postings)
                idf = math.log10(self.total_docs / df) if df > 0 else 0
                stem_idfs.append((stem, idf, postings))
            else:
                stem_idfs.append((stem, 0.0, []))

        # Sort descending by IDF so we can always keep the top term.
        stem_idfs.sort(key=lambda x: x[1], reverse=True)

        # Keep terms above the threshold; always keep at least the highest-IDF term.
        filtered = []
        postings_cache = {}
        for i, (stem, idf, postings) in enumerate(stem_idfs):
            if idf >= IDF_THRESHOLD or (not filtered):
                if postings:  # skip terms with zero postings
                    filtered.append(stem)
                    postings_cache[stem] = postings

        print(f"After IDF filtering: keeping {len(filtered)}/{len(query_stems)} stems: {filtered}")
        return filtered, postings_cache

    def _fetch_all_postings(self, query_stems, postings_cache=None):
        """Retrieves postings lists for all stems, using cache to prevent redundant I/O."""
        term_postings_dict = dict(postings_cache) if postings_cache else {}
        for stem in query_stems:
            if stem in term_postings_dict:
                continue
            postings = self._fetch_postings(stem)
            if not postings:
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
        stems them, applies High-IDF index elimination (Heuristic 1), fetches
        matching posting records from disk, evaluates the multi-term intersection
        with Soft Conjunction fallback (Heuristic 2), ranks findings via TF-IDF
        with Proximity Scoring (Heuristic 3), and resolves the top 5 DocIDs to URLs.
        """
        query_stems = self._get_query_stems(query_str)
        if not query_stems:
            return []

        # HEURISTIC 1: Drop low-IDF (stop-word) terms. Also fetches postings for
        # the surviving terms as a side effect (cached to avoid double disk reads).
        filtered_stems, postings_cache = self._filter_low_idf_stems(query_stems)

        if not filtered_stems:
            return []

        # Build the full postings dict from cache (no re-fetching).
        term_postings_dict = dict(postings_cache)

        # HEURISTIC 2: Soft Conjunction Fallback (Lecture 22, p. 41-46).
        # Start with strict AND across all filtered stems. If we get fewer than
        # MIN_RESULTS, iteratively drop the lowest-IDF term and retry.
        # This implements the lecture's "3 of 4 terms" iterative fallback.
        active_stems = list(filtered_stems)  # already sorted high-IDF first
        ranked_postings = []

        while active_stems:
            postings_list = [term_postings_dict[s] for s in active_stems]
            matched_postings = self._intersect_postings(active_stems, postings_list)

            if matched_postings:
                ranked_postings = self._rank_documents(
                    matched_postings, active_stems, term_postings_dict
                )

            if len(ranked_postings) >= MIN_RESULTS:
                print(f"AND({len(active_stems)} terms) → {len(ranked_postings)} results. Done.")
                break

            # Not enough results — drop the lowest-IDF term (last in the sorted list)
            # and widen to N-1-of-N matching.
            dropped = active_stems.pop()  # lowest IDF is at the end
            print(f"Soft fallback: dropped '{dropped}', retrying with {active_stems}")

            if not active_stems:
                break

        if not ranked_postings:
            return []

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


    test_queries = [
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

        # LONG QUERY TIME: Search takes ~320 ms
        "school of donald bren computer science data science software engineering statistics informatics club and school events at uci for students to attend",
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
    print("      UCI ICS SEARCH ENGINE - M3 BENCHMARK RUNNER       ")
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