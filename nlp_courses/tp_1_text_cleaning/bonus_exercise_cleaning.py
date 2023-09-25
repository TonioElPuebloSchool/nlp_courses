import re  # noqa: F401
import string  # noqa: F401
# modified by Antoine Courbi during NLP TP1


import nltk  # noqa: F401
import pandas as pd
from nltk.corpus import stopwords, wordnet  # noqa: F401
from nltk.stem import WordNetLemmatizer  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import FunctionTransformer  # noqa: F401
from utils import emojis_unicode, emoticons, slang_words  # noqa: F401

from bs4 import BeautifulSoup
from spellchecker import SpellChecker
nltk.download('stopwords')
from collections import Counter
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

# Declare your cleaning functions here
# Chain those functions together inside the preprocessing pipeline
# You can use (or not) Sklearn pipelines and functionTransformer for readability
# and modularity
# --- Documentation ---
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html


# here are the functions I created to clean the text
# they partly come from the notebook we did in class
# some of them are not used in the pipeline (see bellow)
def lower_case(text: str) -> str:
    """
    Converts the input text to lowercase.
    """
    return text.lower()

def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from the input text.
    """
    PUNCT_TO_REMOVE = string.punctuation
    translation_table = str.maketrans('', '', PUNCT_TO_REMOVE)
    return text.translate(translation_table)

def remove_stopwords(text: str,language: str) -> str:
    """
    Removes stopwords from the input text.
    """
    STOPWORDS = set(stopwords.words(language))
    split = text.split()
    filtered_words = [word for word in split if word not in STOPWORDS]
    return " ".join(filtered_words)
'''
def remove_frequent_words(text: str, freq_words: list) -> str:
    """
    Removes frequent words from the input text.
    """
    most_common = Counter(" ".join(text_df["text_wo_stop"]).split()).most_common()
    FREQWORDS = [w for (w, word_count) in most_common[:10]]
    split = text.split()
    filtered_words = [word for word in split if word not in freq_words]
    return " ".join(filtered_words)
    
Here I found out it was difficult to remove words on the whole dataset 
Because the apply function is applying per row not on the whole
Making it difficult to remove the most common words

def remove_rare_words(text: str, rare_words: list) -> str:
    """
    Removes rare words from the input text.
    """
    split = text.split()
    filtered_words = [word for word in split if word not in rare_words]
    return " ".join(filtered_words)
'''
def stemming(text: str) -> str:
    """
    Applies stemming to words in the input text.
    """
    stemmer = PorterStemmer()
    split = text.split()
    filtered_words = [stemmer.stem(word) for word in split]
    return " ".join(filtered_words)

def lemmatize(text: str) -> str:
    """
    Lemmatizes words in the input text.
    """
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }
    pos_tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
    return " ".join(lemmatized_words)

def convert_emoticons(text: str) -> str:
    """
    Converts emoticons to text in the input text.
    """
    EMOTICONS = emoticons()
    for emoticon, description in EMOTICONS.items():
        text = re.sub(emoticon, "_".join(description.replace(",", "").split()), text)
    return text

def convert_emojis(text: str) -> str:
    """
    Converts emojis to text in the input text.
    """
    EMO_UNICODE = emojis_unicode()
    #UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()}
    for description, emoji in EMO_UNICODE.items():
        text = text.replace(emoji, "_".join(description.replace(",", "").replace(":", "").split()))
    return text

def remove_urls(text: str) -> str:
    """
    Removes URLs from the input text.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_http_tags(text: str) -> str:
    """
    Removes HTTP tags from the input text.
    """
    return BeautifulSoup(text, "html.parser").text

def chat_words_conversion(text: str) -> str:
    """
    Converts chat words to standard words in the input text.
    """
    slang_words_list = slang_words()
    new_text = []
    for word in text.split():
        new_text.append(slang_words_list.get(word.upper(), word))
    return " ".join(new_text)

def spell_correction(text: str) -> str:
    """
    Corrects spelling errors in the input text.
    """
    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word is not None else word)
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# We can test the functions we just created using the following code:
assert lower_case("Hello World!") == "hello world!"
assert remove_punctuation("Hello, World!") == "Hello World"
assert remove_stopwords("Hello the World!", 'english') == "Hello World!"
assert stemming("console consoling") == "consol consol"
assert lemmatize("feet caring") == "foot care"
assert convert_emoticons("I am sad :(") == "I am sad Frown_sad_andry_or_pouting"
assert convert_emojis("game is on 🔥") == "game is on fire"
assert remove_urls("https://www.google.com") == ""
assert remove_http_tags("<p>hello world</p>") == "hello world"
assert chat_words_conversion("one minute BRB") == "one minute Be Right Back"
assert (spell_correction("THISNOTAWORD Hopefully you larned smething durng th classn, seeee you in twwo wekks !")) == "THISNOTAWORD Hopefully you learned something during the class see you in two weeks !"

def preprocessing_pipeline(text: str) -> str:
    """
    Chains all the cleaning functions together using scikit-learn pipelines.
    """
    # the variable are defined within the function to avoid errors
    # the pipelinge is processing the text as follow :
    # lower_case -> 
    # remove_urls ->
    # remove_http_tags ->
    # remove_punctuation ->
    # remove_stopwords ->
    # lemmatize ->
    # convert_emoticons ->
    # convert_emojis ->
    # chat_words_conversion ->
    # spell_correction
    # at the end the text is cleaned and ready to be used for NLP
    
    preprocessing_steps = [
        ('lower_case', FunctionTransformer(lower_case)),
        ('remove_urls', FunctionTransformer(remove_urls)),
        ('remove_http_tags', FunctionTransformer(remove_http_tags)),
        ('remove_punctuation', FunctionTransformer(remove_punctuation)),
        ('remove_stopwords', FunctionTransformer(lambda x: remove_stopwords(x, 'english'))),
        #('remove_frequent_words', FunctionTransformer(lambda x: remove_frequent_words(x))),
        #('remove_rare_words', FunctionTransformer(lambda x: remove_rare_words(x))),
        # stemming is less efficient than lemmatization so we are not using it here
        # ('stemming', FunctionTransformer(lambda x: stemming(x))),  # Replace 'stemmer' with your stemmer object
        ('lemmatize', FunctionTransformer(lambda x: lemmatize(x))),  # Replace 'lemmatizer' with your lemmatizer object
        ('convert_emoticons', FunctionTransformer(lambda x: convert_emoticons(x))),
        ('convert_emojis', FunctionTransformer(lambda x: convert_emojis(x))),
        ('chat_words_conversion', FunctionTransformer(lambda x: chat_words_conversion(x))),  # Replace 'slang_words_list' with your dictionary
        ('spell_correction', FunctionTransformer(lambda x: spell_correction(x)))  # Replace 'spell' with your SpellChecker object
    ]

    # Create the pipeline
    preprocessing_pipeline = Pipeline(preprocessing_steps)

    # Apply the pipeline to the input text
    cleaned_text = preprocessing_pipeline.transform([text][0])

    # retunr the cleaned text
    return cleaned_text



if __name__ == "__main__":
    df = pd.read_csv("nlp_courses/tp_1_text_cleaning/to_clean.csv", index_col=0)
    df["cleaned_text"] = df.text.apply(lambda x: preprocessing_pipeline(x))
    for idx, row in df.iterrows():
        print(f"\nBase text: {row.text}")
        print(f"Cleaned text: {row.cleaned_text}\n")
        
# Here is the output of the cleaned text :

# Base text: Hello Amazon - my package never arrived :( https://www.amazon.com/gp/css/order-history?ref_=nav_orders_first PLEASE FIX ASAP ⏰! @AmazonHelp <test/>
# Cleaned text: hello amazon package never arrive please fix As Soon As Possible alarm_clock amazonhelp


# Base text: Hello! 😊 This is an example text with emojis! 👍
# Cleaned text: hello smiling_face_with_smiling_eyes example text emojis thumbs_up


# Base text: <p>This is a <b>sample</b> text with <a href='https://www.example.com'>HTML</a> tags.</p>
# Cleaned text: sample text ref tags


# Base text: The quick brown fox jumps over the lazy dog.
# Cleaned text: quick brown fox jump lazy dog


# Base text: Visit our website at https://www.example.com for more information
# Cleaned text: visit website information


# Base text: I'm feeling 😄 today. Don't worry 😉.
# Cleaned text: im feel smiling_face_with_open_mouth_&_smiling_eyes today dont worry winking_face


# Base text: This text contains special characters #$%&@*!
# Cleaned text: text contain special character


# Base text: LOL BRB and OMG are common chat abbreviations.
# Cleaned text: Laughing Out Loud Be Right Back Oh My God common chat abbreviation


# Base text: 😂😍👏 Just saw the funniest movie ever! 😂😍👏
# Cleaned text: face_with_tears_of_joysmiling_face_with_heart-eyesclapping_hands saw funny movie ever face_with_tears_of_joysmiling_face_with_heart-eyesclapping_hands


# Base text: <a href='https://www.example.com'>Click here</a> for more info
# Cleaned text: ref here info


# Base text: I found a great recipe at https://www.recipes.com! 😋 It's so delicious! #cooking
# Cleaned text: find great recipe face_savouring_delicious_food delicious cooking


# Base text: M8 imho this NLP thing is kinda 🔥 !
# Cleaned text: Mate In My Humble Opinion nap thing kinda fire

# We can see it worked pretty well, there are fews things I had to change to make it work
# For example the order for the pipeline had to be reviewed
# The punctuation can't be placed before the url removal for example
# Also it's difficult to remove the most common words because the apply function is applying per row not on the whole
