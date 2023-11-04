# TinyStoriesV2 Dataset - Normalized

Welcome to a normalized version of the TinyStoriesV2-GPT4 Dataset. This is a processed version of the original dataset tailored for use with language models and other natural language processing applications. The data has been normalized and filtered for even more simplicity, so as make training toy language models easier without having to worry about unknown tokens

## Dataset Description

The TinyStoriesV2 dataset is a collection of extremely simple narrative texts ideal for training and evaluating the capabilities of small language models. In this section, the data has been preprocessed with normalization functions to maintain consistency in punctuation and character use, ensuring a clean and uniform dataset that minimizes noise and improves model performance.

## Normalization Process

The normalization process involves several steps, including whitespace normalization, quote standardization, ASCII character conversion, ellipsis replacement, and accent removal. This ensures the data is in a consistent format that is easily interpretable by NLP algorithms.

### Functions Used for Normalization

1. `normalize(text: str) -> str`: This function normalizes whitespace, replaces special quote characters with ASCII equivalents, converts Windows-1252 characters to ASCII, replaces ellipses with three periods, and removes diacritics from characters.

2. `contains_only_allowed_chars(text: str) -> bool`: This function checks the text to ensure it contains only allowed characters, which include alphanumeric characters and basic punctuation (`. , ? ! ' "`).

## Dataset Structure

The dataset has been chunked into batches of 100,000 entries for ease of use. Each entry has been processed through the normalization functions to ensure consistency and quality. There is 1 chunk in the valid split and 25 in the train split

## Usage

This dataset is shared under the CDLA-Sharing-1.0 license, allowing for free use with appropriate attribution.

To use this dataset, please cite as:

```bibtex
@misc{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Ronen Eldan and Yuanzhi Li},
  year={2023},
  eprint={2305.07759},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

For more information, please refer to the original HuggingFace repository: [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)
You can also read the related research paper on [arXiv](https://arxiv.org/abs/2305.07759).

### Functions

```python
import unicodedata

def normalize(text: str) -> str:
    """
    Normalize the input text

    >>> Encoder.normalize('Hëllo  world! ')
    'Hello world!'
    """

    # Normalize whitespace, both between words and on the end
    text = " ".join(text.split())

    # Replace curly double quotes with straight double quotes
    text = text.replace('“', '"').replace('”', '"')

    # Replace curly single quotes with straight single quotes
    text = text.replace("‘", "'").replace("’", "'")

    # Replace Windows-1252 single quotes with ASCII single quote
    text = text.replace('\x92', "'")
    # Replace Windows-1252 opening double quotes with ASCII double quote
    text = text.replace('\x93', '"')
    # Replace Windows-1252 closing double quotes with ASCII double quote
    text = text.replace('\x94', '"')

    # replace ellipsis character with three periods
    text = text.replace("…", "...")

    # replace backtick with apostrophe
    text = text.replace("`", "'")

    # e.g., converting "ë" to "e" and "‼" to "!" for NFD normalization.
    text = "".join(
        (
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    )

    return text
```

```python
def contains_only_allowed_chars(text: str) -> bool:
    """
    Check that a text contains only alphanumeric chars and basic punctuation
    """

    alphanumeric = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    punctuation = " .,?!'\""

    allowlist = set(alphanumeric + punctuation)
    for char in text:
        if char not in allowlist:
            print(f"Invalid character found: {char}")
            return False
    return True
```