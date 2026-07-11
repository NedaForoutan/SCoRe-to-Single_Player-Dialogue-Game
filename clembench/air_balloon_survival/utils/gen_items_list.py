import nltk
from nltk.corpus import wordnet as wn, brown, europarl_raw
from nltk import FreqDist, pos_tag, word_tokenize
import pandas as pd
import json
import random
import os

# THIS FILE STILL NEEDS UPDATES AS THE WORDS GENERATED (ESPECIALLY IN THE GERMAN VERSION ARE NOT EVERYDAY OBJECTS)
# WE WILL EITHER DO THIS OR DROP THIS EXPERIMENT TYPE AND EXPAND THE OTHER TWO TYPES

LANGUAGE='en'

# Ensure required NLTK resources are downloaded
nltk.download('brown', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('universal_tagset', quiet=True)
nltk.download('europarl_raw', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Silence NLTK warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nltk.corpus.reader.wordnet")

# Define inclusion categories
ALLOWED_ROOTS = [
    wn.synset("artifact.n.01"), #man-made objects
    #wn.synset("natural_object.n.01"), #naturally occuring objects
    wn.synset("food.n.01"),
    #wn.synset("implement.n.01"), # tools, utensils, etc. 
    #wn.synset("container.n.01"), # objects that can hold things
    #wn.synset("instrumentality.n.03") # generic tools
]

# Define exclusion categories
BANNED_ROOTS = [
    wn.synset("abstraction.n.06"), #conctpts, ideas
    wn.synset("person.n.01"), 
    wn.synset("location.n.01"), # geographical locations
    wn.synset("institution.n.01"), #organisations
    wn.synset("communication.n.02"), # messages, words, expressions
    wn.synset("group.n.01"), # collections of people or things
]


def physical_item(word):
    """Return True if word belongs to a concrete, game-useful physical item category as defined."""
    for syn in wn.synsets(word, pos=wn.NOUN):
        for root in ALLOWED_ROOTS:
            if root in syn.closure(lambda s: s.hypernyms()):
                return True
    return False


def blacklisted_item(word):
    """Return True if word belongs to abstract, institutional, or unwanted concepts as defined."""
    for syn in wn.synsets(word, pos=wn.NOUN):
        for root in BANNED_ROOTS:
            if root in syn.closure(lambda s: s.hypernyms()):
                return True
    return False


def extract_physical_nouns(language, n_words, min_len):
    """Extract physical nouns depending on the language."""

    # TODO: still needs adjustement to actually only generate physical items (from survival context etc.)
    if language == 'en':
        tagged_words = brown.tagged_words(tagset='universal')
        nouns = [word.lower() for word, tag in tagged_words if tag == 'NOUN' and word.isalpha() and len(word) >= min_len]

        fdist = FreqDist(nouns)
        sorted_nouns = sorted(fdist.items(), key=lambda x: x[1], reverse=True)

        final_physical = []
        for word, count in sorted_nouns:
            if physical_item(word) and not blacklisted_item(word):
                final_physical.append(word)
            if len(final_physical) >= n_words:
                break

        return final_physical
    
    # TODO: still needs adjustement to actually only generate physical items (from survival context etc.)
    elif language == 'de':
        text = europarl_raw.german.raw()
        tokens = word_tokenize(text, language='german')
        words = [word.lower() for word in tokens if word.isalpha() and len(word) >= min_len]
        fdist = FreqDist(words)
        return [word for word, _ in fdist.most_common(n_words)]


    else:
        raise ValueError(f"Unsupported language: {language}")

def generate_random_words(letters, n_words, min_len, max_len):
    """
    Generates a list of randomly generated words based on given input
    Paremeters:
    letters (str): list of strings as the input
    n_words (int): number of words to generate
    min_len (int): minimum length of the words
    max_len (int): maximum length of the words
    Returns:
    list: a list of randomly generated, unique and lowercase words
    """
    generated = set()
    attempts = 0
    max_attempts = n_words * 10 #set a limit to avoid infinite loop

    while len(generated) < n_words and attempts < max_attempts:
        word_length = random.randint(min_len, max_len)
        word = ''.join(random.choices(letters, k=word_length))

        #append digits randomly to the word
        # num_digits = random.choice([1, 2, 3]) #attach 1-3 digits
        digits = ''.join(random.choices('0123456789', k=2))
        word += digits
        
        generated.add(word)
        attempts += 1
         
    return list(generated)

if __name__ == "__main__":
    common_nouns = extract_physical_nouns(LANGUAGE, 500, 4) #number of words and min_length
    letters = list('ABC')
    generated_words = generate_random_words(letters, 500, 1, 1) #letters, n_words, min_len, max_len

    output = {
        "common_nouns": common_nouns,
        "generated_words": generated_words
    }

    # Adjust reusability with different language - TODO
    output_dir = f"air_balloon_survival/resources/{LANGUAGE}/"
    output_path = os.path.join(output_dir, "items.json")


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
