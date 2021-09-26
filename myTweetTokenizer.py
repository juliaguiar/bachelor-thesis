import spacy
from spacy.tokens import Doc
import twikenizer as twk

twk = twk.Twikenizer()

class MyTweetTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tweet):
        words = twk.tokenize(tweet)
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
