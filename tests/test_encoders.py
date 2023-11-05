import pytest

from encoders.byte_pair_encoder import BytePairEncoder


@pytest.fixture
def encoder():
    return BytePairEncoder()


test_cases = [
    (
        ["This is a test.", "This is also a test."],
        {
            "This|": 2,
            "is|": 2,
            "a|": 2,
            "test|": 2,
            "also|": 1,
            ".": 2,
        },
    ),
    (  # split contractions on the apostrophe
        ["It's time to go.", "It is time to go."],
        {
            "It|": 1,
            "'s|": 1,
            "It": 1,
            "is|": 1,
            "time|": 2,
            "to|": 2,
            "go|": 2,
            ".": 2,
        },
    ),
    (
        ['She said "hello."'],
        {"She|": 1, "said|": 1, "hello|": 1, ".": 1, '"': 2},
    ),
]


@pytest.mark.usefixtures("encoder")
class TestBytePairEncoder:
    @pytest.mark.parametrize("test_input,expected", test_cases)
    def test_get_word_frequencies(self, encoder, test_input, expected):
        frequencies = encoder.get_word_frequencies(test_input)
        assert frequencies == expected

    def test_build_initial_vocab(self, encoder):
        frequencies = {
            "This|": 2,
            "is|": 2,
            "a|": 2,
            "test|": 2,
            "also|": 1,
            ".": 2,
            "It": 2,
            "'s|": 1,
        }

        vocab = encoder.build_initial_vocab(frequencies)

        expected = ["'", ".", "I", "T", "a", "e", "h", "i", "l", "o", "s", "t", "|"]

        assert vocab == expected
