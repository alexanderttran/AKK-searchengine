import json
import os
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def process_content(html_content):
    soup = BeautifulSoup(html_content, 'lxml')

    #look for important stems
    important_tags = ['title', 'h1', 'h2', 'h3', 'b', 'strong']
    important_stems = set()
    for tag in soup.find_all(important_tags):
        for token in re.findall(r'[a-zA-Z0-9]+', tag.get_text()):
            important_stems.add(stemmer.stem(token))

    #pass over entire document
    all_tokens = re.findall(r'[a-zA-Z0-9]+', soup.get_text().lower())

    #stem and count
    token_counts = {}
    for token in all_tokens:
        stemmed = stemmer.stem(token)
        if stemmed not in token_counts:
            is_important = 1 if stemmed in important_stems else 0
            token_counts[stemmed] = [1, is_important]
        else:
            if token in important_stems:
                token_counts[stemmed][1] = 1

            token_counts[stemmed][0] += 1

    return token_counts