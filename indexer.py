import pickle

def save_index_to_disk(index_to_save, filename):
    #save inverted index to a binary file because its faster than json

    #sort by token because needed for later merge
    sorted_index = dict(sorted(index_to_save.items()))

    with open(filename, 'wb') as f:
        pickle.dump(sorted_index, f)
    print(f"Index saved to {filename}")