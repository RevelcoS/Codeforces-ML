import nltk
from nltk.corpus import stopwords as _stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from tqdm import tqdm
import string

class Resources:

    punctuations = None
    stopwords = None
    stemmer = None
    lemmatizer = None

    @staticmethod
    def download():

        # TODO: Add download dataset from keggle
        # Note: Should NOT do it here.
        #       Better do database and just add problems from keggle in it.

        # Download NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('wordnet')

    @staticmethod
    def load():

        # Unpack NLTK resources
        Resources.punctuations = set(string.punctuation)
        Resources.stopwords = set(_stopwords.words('english'))
        Resources.stemmer = PorterStemmer()
        Resources.lemmatizer = WordNetLemmatizer()

        # Use tqdm pandas extension for progress_apply function
        tqdm.pandas()
