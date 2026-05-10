from collections import defaultdict
import os
import json

from parser_utils import process_content

inverted_index = defaultdict(list)
url_to_id_map = {}
doc_count = 0

#path to temp testing files
data_dir = '../developer/DEV_SMALL'

def get_doc_id(url, url_to_id_map):
    if url not in url_to_id_map:
        new_id = len(url_to_id_map)
        url_to_id_map[url] = new_id
        return new_id, True
    return url_to_id_map[url], False

for root, dirs, files in os.walk(data_dir):
    print('Found directory', root)

    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                url = data['url']
                content = data['content']

                doc_id, is_new_id = get_doc_id(url, url_to_id_map)
                doc_count += 1

                #process the document
                stats = process_content(content)

                for token, (term_freq, importance) in stats.items():
                    inverted_index[token].append((doc_id, term_freq, importance))

with open("url_map.json", 'w', encoding='utf-8') as f:
    json.dump(url_to_id_map, f)
print(f"Saved {len(url_to_id_map)} URL map")

from indexer import save_index_to_disk
save_index_to_disk(inverted_index, "index_part_1.bin")

print(f"\n--- M1 Analytics for {doc_count} Docs ---")
print(f"Total Documents: {doc_count}")
print(f"Unique Tokens: {len(inverted_index)}")

