"""Assorted utitilies for Lightning plus Weights & Biases."""
import random
import string
import warnings

from pytorch_lightning.utilities.warnings import LightningDeprecationWarning


try:
    from wonderwords import RandomWord
    no_wonderwords = False
except ImportError:
    no_wonderwords = True


if no_wonderwords:
    chars = string.ascii_lowercase

    def make_random_name():
        return "".join([random.choice(chars) for ii in range(10)])

else:
    r = RandomWord()

    def make_random_name():
        name = "-".join(
            [r.word(word_min_length=3, word_max_length=7, include_parts_of_speech=["adjective"]),
             r.word(word_min_length=5, word_max_length=7, include_parts_of_speech=["noun"])])
        return name


def filter_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=LightningDeprecationWarning)
