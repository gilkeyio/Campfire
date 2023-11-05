from collections import defaultdict
from typing import List, Dict, Tuple
from data.loader import load_dataset


class BytePairEncoder:
    def __init__(self):
        pass

    def get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        vocab = defaultdict(int)
        punctuation = ".,?!'\""

        # Define end-of-word token
        eow = "|"

        for text in corpus:
            for word in text.split():
                # seperate leading punctuation
                while word and word[0] in punctuation:
                    vocab[word[0]] += 1
                    word = word[1:]

                # seperate trailing punctuation
                while word and word[-1] in punctuation:
                    vocab[word[-1]] += 1
                    word = word[:-1]

                # Check for contractions and handle accordingly
                if "'" in word:
                    # Separate the contraction part and the rest of the word
                    parts = word.split("'")
                    base_word = parts[0]
                    contraction = "'" + parts[1]

                    vocab[base_word] += 1
                    vocab[contraction + eow] += 1
                else:
                    # Split the word into characters and add end-of-word token
                    vocab[word + eow] += 1

        return dict(vocab)

    def build_initial_vocab(self, word_frequencies: Dict[str, int]) -> List[str]:
        vocab = []

        for word in word_frequencies.keys():
            for character in word:
                if character not in vocab:
                    vocab.append(character)
        vocab.sort()

        return vocab

    def get_most_frequent_pair():
        pass


dataset = load_dataset()

encoder = BytePairEncoder()

freq = encoder.get_word_frequencies(dataset["train"][:100])

print(encoder.build_initial_vocab(freq))
