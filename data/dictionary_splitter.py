from functional import seq
import json
import os
import pickle
import re
from sklearn.model_selection import train_test_split

def split_dictionary_entry(entry):
    """
    Splits each dictionary entry into its parts by numbers. Also splits using semicolons.
    """

    def split_paragraphs(x):
        return re.split("\n\n", x)

    def definition_heads(x):
        return re.split(r'\s*\d\..',x) if re.match(r'\s*\d\..*',x) else [x]

    def split_sentences(x):
        return re.split(r"(?<!\d)\.[\s$]", x)[0]

    def split_synonyms(x):
        return re.split(r";", x)

    return seq(entry)\
        .flat_map(split_paragraphs)\
        .flat_map(definition_heads)\
        .map(split_sentences)\
        .flat_map(split_synonyms)\
        .filter(lambda x: x is not '')\
        .map(lambda x: x.strip())

def generate_un_prefixed_word_dictionaries(dictionary):
    """
    Returns dictionaries containing words that have corresponding words with un- prefixed.
    """
    negative_words = {key: value for key, value in dictionary.items() if key[:2] == 'un'}
    positive_words = {word[2:]:dictionary[word[2:]] for word in negative_words if word[2:] in dictionary}
    negative_words = {key:value for key, value in negative_words.items() if key[2:] in positive_words}

    negative_words = {key: split_dictionary_entry(value) for key, value in negative_words.items()}
    positive_words = {key: split_dictionary_entry(value) for key, value in positive_words.items()}

    # print(positive_words['clean'])
    # print(negative_words['unclean'])
    return positive_words, negative_words


def train_test_validate_split(dictionary, folder="ptb"):
    words = list(dictionary.keys())
    definitions = list(dictionary.values())

    train_words, test_words, train_definitions, test_definitions = train_test_split(words, definitions, test_size=0.1, random_state=42)

    train_words, valid_words, train_definitions, valid_definitions = train_test_split(train_words, train_definitions, test_size=0.1, random_state=42)

    def processed_word(word):
        return word + "\n"
        # return " ".join(word) + "\n"

    if not os.path.exists(folder):
        os.makedirs(folder)

    for type, dataset in zip(['train', 'test', 'valid'], [train_definitions, test_definitions, valid_definitions]):
            with open(os.path.join(folder, "{}.txt".format(type)), 'w') as f:
                for word in dataset:
                    f.write(processed_word(word))

    print("TRAIN: {} VALID: {} TEST: {}".format(len(train_words),
                                                len(valid_words),
                                                len(test_words)
                                                ))

with open('WebstersEnglishDictionary/dictionary.json') as f:
    dictionary = json.load(f)
    print("Dictionary size: {}".format(len(dictionary)))

pos, neg = generate_un_prefixed_word_dictionaries(dictionary)

tuples = {word: definition for dict in [pos, neg] for word, definitions in dict.items() for definition in definitions[:1]}

first_def = {word: definition for word, definitions in dictionary.items() for definition in split_dictionary_entry(definitions)[:1]}

train_test_validate_split(first_def, folder="first_def")
