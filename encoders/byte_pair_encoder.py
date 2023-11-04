from collections import defaultdict
from typing import List, Dict, Tuple


class BytePairEncoder:
    def build_initial_vocab(
        self, texts: List[str], special_tokens: List[str] = []
    ) -> Dict[str, int]:
        vocab = defaultdict(int)

        # Define punctuation characters to be tokenized separately
        punctuations = ".,?!'-\""

        for text in texts:
            words = []
            for word in text.split():
                # If the word is a special token, treat it atomically
                if word in special_tokens:
                    words.append(word)
                    continue

                # Split word if it starts with punctuation
                while word and word[0] in punctuations:
                    words.append(word[0])
                    word = word[1:]

                # Split word if it ends with punctuation, ensuring all punctuations are separate tokens
                while word and word[-1] in punctuations:
                    words.append(word[-1])
                    word = word[:-1]
                if word:
                    words.append(word)

            # Tokenize each word or punctuation in our modified list
            for word in words:
                if len(word) == 1 and word in punctuations:
                    vocab[word] += 1
                else:
                    word = " ".join(list(word))
                    vocab[word] += 1

        return vocab

    def get_most_common_pair(self, vocab: Dict[str, int]) -> Tuple[str, str]:
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            tokens = word.split()

            # Count the occurrences of pairs of tokens
            for i in range(len(tokens) - 1):
                pairs[tokens[i], tokens[i + 1]] += freq

        # Return the most common pair
        return max(pairs, key=pairs.get)

    def merge_most_common(
        self, vocab: Dict[str, int], token_pair: Tuple[str, str]
    ) -> Dict[str, int]:
        new_vocab = defaultdict(int)
        bigram = " ".join(token_pair)
        replacement = "".join(token_pair)

        for word, count in vocab.items():
            # Replace all instances of the most common pair with the merged token
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = count

        return new_vocab

    def bpe_merge(
        self, initial_vocab: Dict[str, int], num_merges: int
    ) -> Dict[str, int]:
        vocab = initial_vocab.copy()

        for i in range(num_merges):
            most_common_pair = self.get_most_common_pair(vocab)
            if not most_common_pair:
                break
            vocab = self.merge_most_common(vocab, most_common_pair)

            print(f"Iteration {i + 1}: Merged {most_common_pair}")

        return vocab


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    dataset = load_from_disk("TinyStories")

    encoder = BytePairEncoder()

    vocab = encoder.build_initial_vocab(dataset["train"]["text"])

    num_iterations = 100

    new_vocab = encoder.bpe_merge(vocab, num_iterations)

    print(new_vocab)

    # dataset = load_dataset("roneneldan/TinyStories")

    # dataset = dataset.filter(
    #     lambda example: Encoder.contains_only_allowed_chars(example["text"])
    # )

    # dataset = dataset.map(lambda example: {"text": Encoder.normalize(example["text"])})

    # dataset.save_to_disk("TinyStories")
