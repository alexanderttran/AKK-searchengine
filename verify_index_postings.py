import pickle


def verify_index_postings(test_word):
    try:
        with open("index_part_1.bin", "rb") as f:
            index = pickle.load(f)

        if test_word in index:
            print(f"Postings for '{test_word}': {index[test_word]}")
        else:
            print(f"Token '{test_word}' not found. Check tokenizer or test data.")

    except FileNotFoundError:
        print("Error: index_part_1.bin not found")

if __name__ == "__main__":
    verify_index_postings("research")
