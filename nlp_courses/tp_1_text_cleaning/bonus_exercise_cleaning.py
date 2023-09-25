import re  # noqa: F401
import string  # noqa: F401

import nltk  # noqa: F401
import pandas as pd
from nltk.corpus import stopwords, wordnet  # noqa: F401
from nltk.stem import WordNetLemmatizer  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import FunctionTransformer  # noqa: F401
from utils import emojis_unicode, emoticons, slang_words  # noqa: F401

# Declare your cleaning functions here
# Chain those functions together inside the preprocessing pipeline
# You can use (or not) Sklearn pipelines and functionTransformer for readability
# and modularity
# --- Documentation ---
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html


def preprocessing_pipeline(text: str) -> str:
    """
    This function should chain all the cleaning functions together
    """ 
        



if __name__ == "__main__":
    df = pd.read_csv("nlp_courses/tp_1_text_cleaning/to_clean.csv", index_col=0)
    df["cleaned_text"] = df.text.apply(lambda x: preprocessing_pipeline(x))
    for idx, row in df.iterrows():
        print(f"\nBase text: {row.text}")
        print(f"Cleaned text: {row.cleaned_text}\n")
