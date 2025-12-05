import nltk
from nltk.corpus import stopwords as _stopwords

from tqdm import tqdm
import string

class Resources:

    punctuations = None
    stopwords = None

    @staticmethod
    def download():

        # TODO: add download dataset from keggle

        # Download NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab') 

    @staticmethod
    def setup():

        # Unpack NLTK resources
        Resources.punctuations = set(string.punctuation)
        Resources.stopwords = set(_stopwords.words('english'))

        # Use tqdm pandas extension for progress_apply function
        tqdm.pandas()


if __name__ == '__main__':
    Resources.download()
