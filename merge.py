"""
UTILITY: K-Way Index Merger & Lexicon Builder (Milestone 2)
---------------------------------------------------------
TEAM BREAKDOWN (Kenrich, Alexander, Kenny):
This module consolidates our sorted partial index batches into a single 
unified text index on disk. It constructs a memory-resident 'lexicon' map 
storing absolute physical byte positions, enabling sub-millisecond O(1) 
disk-seeking when we execute multi-word queries in search.py.
"""

import pickle
import json
import heapq
from pathlib import Path
from itertools import groupby

# Global configuration paths for the inverted index infrastructure
BATCH_FOLDER = Path("index_batches")
GLOBAL_INDEX_PATH = Path("global_index.txt")
LEXICON_PATH = Path("lexicon.json")


def merge_indexes():
    # TEAM NOTE: We MUST sort the file paths chronologically. 
    # Because main.py processes documents sequentially, sorting the files ensures 
    # that our lowest DocIDs (part 1) enter the merge pipeline before higher DocIDs (part 2).
    index_files = sorted(BATCH_FOLDER.glob("index_part_*.bin"))

    if not index_files:
        print("No partial index files found!")
        return

    # Load all partial index dictionaries into RAM for this offline merge phase
    loaded_indexes = []
    for file_path in index_files:
        with open(file_path, 'rb') as f:
            loaded_indexes.append(pickle.load(f))

    # TEAM NOTE: Convert dictionary views into stateful Python Iterators using iter().
    # This prevents us from copying or joining dictionaries directly in memory. 
    # Each iterator maintains an internal pointer to its front-most alphabetical key.
    iterators = [iter(idx.items()) for idx in loaded_indexes]

    print("Iterators prepared. Ready to merge.")

    # TEAM NOTE: The Asterisk (*) acts as the positional argument unpacking ("splat") operator.
    # It unrolls our list bundle, feeding all independent streams into heapq.merge simultaneously.
    # Because heapq.merge is a stable merge, it preserves chronological input stream order during ties, 
    # and the lambda function guarantees a global alphabetical traversal sorted by token (index 0).
    merged_stream = heapq.merge(*iterators, key=lambda item: item[0])

    # Dictionary to hold our structural coordinate mapping data
    lexicon = {}

    print("Executing linear merge and streaming to global_index.txt")

    # TEAM NOTE: We use Binary Write mode ('wb') explicitly instead of plain text mode ('w').
    # This completely bypasses automated OS text adaptations (e.g., converting \n to \r\n on Windows).
    # This ensures a byte-for-byte uniform file structure across different development environments,
    # keeping our coordinate offsets perfectly aligned for everyone on the team.
    with open(GLOBAL_INDEX_PATH, 'wb') as f_out:
        # groupby identifies consecutive matching tokens streaming out of our sorted min-heap pipeline
        for token, group in groupby(merged_stream, key=lambda item: item[0]):
            combined_postings = []

            # Unpack the active group stream elements (e.g., ("machin", [postings]))
            for token_str, postings_list in group:
                # TEAM NOTE: Since files entered the pipeline chronologically sorted, 
                # a simple .extend() seamlessly stitches postings blocks together back-to-back 
                # while strictly preserving perfectly ascending DocID order inside the final list.
                combined_postings.extend(postings_list)

            # TEAM NOTE: f_out.tell() captures the EXACT absolute physical byte offset 
            # where the line begins on the hard drive BEFORE writing the token string. 
            # This is NOT a line number; it is a direct pointer location for file.seek() at query time.
            lexicon[token] = f_out.tell()

            # Format the line database row layout
            dataToWrite = f"{token}|{json.dumps(combined_postings)}\n"

            # Convert human-readable strings into raw binary bits to write to a 'wb' file stream
            f_out.write(dataToWrite.encode('utf-8'))

    # TEAM NOTE (TODO next session): Remember to save the 'lexicon' dictionary to disk 
    # using json.dump(lexicon, f) here so search.py can open it!


if __name__ == "__main__":
    merge_indexes()