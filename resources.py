import nltk
from nltk.corpus import stopwords as _stopwords

import string

class Resources:

    punctuations = None
    stopwords = None

resources = Resources()

def download():

    # TODO: add download dataset from keggle

    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab') 

def unpack():

    # Unpack NLTK resources
    global resources 
    resources.punctuations = set(string.punctuation)
    resources.stopwords = set(_stopwords.words('english'))

if __name__ == '__main__':
    download()
