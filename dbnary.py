import csv
import re
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, OWL
from dictionary_splitter import train_test_validate_split

dbnary = Namespace("http://kaiko.getalp.org/dbnary#")
dbnary_eng = Namespace("http://kaiko.getalp.org/dbnary/eng/")
lemon = Namespace("https://lemon-model.net/lemon#")
lexinfo = Namespace("https://www.lexinfo.net/ontology/2.0/lexinfo#")
ontolex = Namespace("http://www.w3.org/ns/lemon/ontolex#")
skos = Namespace("http://www.w3.org/2004/02/skos/core#")

def get_all_senses(word, g):
    canonicalForm = g.value(subject=word, predicate=ontolex.canonicalForm)
    word_string = g.value(subject=canonicalForm, predicate=ontolex.writtenRep)

    def_pairs = []
    for sense in g.objects(subject=word, predicate=ontolex.sense):
        definition = g.value(subject=sense, predicate=skos.definition)
        definition_string = g.value(subject=definition, predicate=RDF.value)
        # make sure each definition string is on a single line
        definition_string = definition_string.replace("\n", " ")
        def_pairs.append((word_string, definition_string))
    return def_pairs

g = Graph()
g.parse("data/en_dbnary_ontolex.ttl", format="turtle")
print ("Parsed Ontolex graph; ready to get definitions")

pairs = []

# TEN THOUSAND WORDS
# with open('data/tenthousandwords.txt', 'r') as f:
#     ten_thousand = f.read().split("\n")
#
# for common_word in ten_thousand:
#     for word in g.objects(subject=dbnary_eng[common_word], predicate=dbnary.describes):
#         # if not meets_conditions(word_string):
#         #     continue
#         # get POS, etc
#         all_senses = get_all_senses(word, g)
#         pairs += all_senses
# train_test_validate_split(pairs, folder="data/dbnary_tenthousand")

# ALL WORDS
# for word in g.subjects(predicate=RDF.type, object=ontolex.LexicalEntry):
#     pairs += get_all_senses(word, g)
# train_test_validate_split(pairs, folder="data/dbnary_full")

# MORPHOLEX
with open('data/word_to_morph.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    word_to_morph = []
    for (index, word, morph) in csv_reader:
        word_to_morph.append((word, morph))

for (word, morph) in word_to_morph:
    tokenized_morph = re.sub("[{(<>)}]", " ", morph)
    for word in g.objects(subject=dbnary_eng[word], predicate=dbnary.describes):
        # if not meets_conditions(word_string):
        #     continue
        # get POS, etc
        all_senses = get_all_senses(word, g)
        all_senses_switched = [(tokenized_morph, definition_string) for (_, definition_string) in all_senses]
        pairs += all_senses_switched
train_test_validate_split(pairs, folder="data/dbnary_morph")



def meets_conditions(word_string):
    return len(word_string) > 3 \
        and len(word_string) < 15\
        and word_string.isalpha()\
        and word_string.islower()\
        and word_string in ten_thousand
