import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Create a stemmer
stemmer = PorterStemmer()


def tokenizer(sentence):
    """
    This function takes a sentence as input and splits it into an array of words or tokens.
    A token can be a word, punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)


def stemer(word):
    """
    This function implements stemming which is the process of reducing a word to its root form.
    For example, "organize", "organizes", and "organizing" would all be reduced to "organ".
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    This function returns a bag of words array. For each known word that exists in the sentence,
    the function places a 1 in the corresponding position of the array and 0 otherwise.
    """
    stemmed_words = [stemer(word) for word in tokenized_sentence]
    word_bag = np.empty(len(all_words), dtype=np.float32)
    word_bag.fill(0.0)

    for index, word in enumerate(all_words):
        if word in stemmed_words:
            word_bag[index] = 1
        else:
            word_bag[index] = 0
    return word_bag
