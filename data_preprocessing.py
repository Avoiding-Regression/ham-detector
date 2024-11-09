from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from num2words import num2words
import re
import contractions
from sklearn.feature_extraction.text import CountVectorizer

class data_preprocessing:
    def __init__(self,text_df):
        self.text_df = text_df
    
    def text_lower(self)->str:
        """
        Lowers all text within the word:

        Args:
            words (str): The input is a word that may have any sort of capitalization
        Returns:
            str: The same word will be returned but it will all be in lower case
        """
        return self.text_df.lower()

    def remove_punctuation(self)->str:
        """
        Removes any puncutations from the text data 

        Args:
            text (str): The input text that needs to have punctuations removed
        Returns:
            str: The same text with removed punctuations
        """
        return self.text_df.translate(str.maketrans('','', string.punctuation))
    
    def stopword_removal(self)->list:
        """
        This function will remove any stopwords from the english language from a tokenized list
        
        Arg (list): 
            Token of a list of words to have the stop words be removed
        Return:
            Returns the list of tokens with stop words removed

        """
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in self.text_df if word not in stop_words]
        return tokens 
    
    def stem_lem(self)->list:
        """
        Conducts stemming and lemmatization on tokens to ensure word precise word definitions
        Arg (list): 
            Tokenized list of words to be lemmed and stemmed
        Return:
            Returns the list of tokens with stemmed and lemmed words
        """

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        tokens = [stemmer.stem(word) for word in self.text_df]
        tokens = [lemmatizer.lemmatize(word) for word in self.text_df]
        return tokens
    
    def convert_tokens_to_words(self)-> list:
        """
        Convert any numeric tokens in a list of tokens to their word equivalents using num2words.
        
        Args:
            tokens (list): A list of tokenized strings.
            
        Returns:
            list: A list with numeric tokens converted to words.
        """
        
        # Convert each token
        converted_tokens = [num2words(int(token)) if token.isdigit() else token for token in self.text_df]
        
        return converted_tokens
    

    def remove_whitespace(self)->list:
        """
        removes any white space within the token any numeric tokens in a list of tokens to their word equivalents using num2words.
        
        Args:
            tokens (list): A list of tokenized strings.
            
        Returns:
            list: A list wtokens with extra white space removed.
        """
        cleaned_tokens = [token.strip() for token in self.text_df]

        return cleaned_tokens
    

    
    def remove_specialchar(self)->list:
        """
        removes any white space within the token any numeric tokens in a list of tokens to their word equivalents using num2words.
        
        Args:
            tokens (list): A list of tokenized strings.
            
        Returns:
            list: A list tokens with special characters removed.

        """
        cleaned_tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in self.text_df]

        return cleaned_tokens
    
    def preprocessing(self):
        """Executes all preprocessing steps"""
        self.text_lower()
        self.remove_punctuation()
        self.stopword_removal()
        self.stem_lem()
        self.convert_tokens_to_words()
        self.remove_whitespace()
        self.remove_specialchar()

class ngrams:
    def get_ngrams(text, n=2):
        """
        Tokenizes the input text and generates n-grams of the specified size.
        
        Args:
            text (str): The input text to be tokenized and transformed into n-grams.
            n (int): The number of tokens to include in each n-gram (default is 2, meaning bigrams).

        Returns:
            list: A list of n-grams, where each n-gram is a string of `n` consecutive tokens. Only n-grams
            containing alphabetic characters are included (e.g., no numbers or special characters).
            
        """
        # Ensure the input is a string
        if not isinstance(text, str):
            return []
        
        # Tokenize and create n-grams
        tokens = word_tokenize(text.lower())  # This is where 'punkt' is needed
        n_grams = ngrams(tokens, n)
        
        # Return n-grams that only contain alphabetic tokens
        return [' '.join(grams) for grams in n_grams if all(word.isalpha() for word in grams)]
