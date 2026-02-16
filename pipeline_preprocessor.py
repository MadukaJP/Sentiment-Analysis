#!/usr/bin/env python
# coding: utf-8

# In[69]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re
import spacy


# In[70]:


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# In[71]:


def normalize_text_ml(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)      # remove HTML first
    text = re.sub(r'[^\w\s]', '', text)    # remove punctuation
    return text


# In[72]:


def handle_negation(tokens):
    negation_words = {"not", "no", "never"}
    negated = False
    new_tokens = []

    for token in tokens:
        if token in negation_words:
            negated = True
            new_tokens.append(token)
            continue
        if re.match(r'[.!?]', token):
            negated = False  # reset at punctuation
        new_tokens.append(token + "_NEG" if negated else token)
    return new_tokens


# In[73]:


stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]


# In[74]:


stemmer = PorterStemmer()


# In[75]:


def preprocess_for_ml(text):
    text = normalize_text_ml(text)
    tokens = [t for t in word_tokenize(text) if t not in stop_words]
    # optional: handle negation
    tokens = handle_negation(tokens)
    stemmed_text = " ".join([stemmer.stem(t) for t in tokens])
    return stemmed_text


# In[76]:


def pos_tag_tokens(tokens):
    return nltk.pos_tag(tokens)


# In[77]:


nlp = spacy.load("en_core_web_sm")


# In[78]:


def extract_entities(text):
    """
    Extract named entities from raw text.

    Why raw text?
    - Capitalization matters
    - Context matters
    """
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "entity_text": ent.text,
            "entity_label": ent.label_
        })

    return entities


# In[80]:


def preprocess_pipeline(text):
    normalized = normalize_text(text)
    tokens = word_tokenize(normalized)
    clean_tokens = remove_stopwords(tokens)
    cleaned_text = preprocess_for_ml(text)
    pos_tags = pos_tag_tokens(clean_tokens)
    entities = extract_entities(text)

    return {
        "raw_text": text,
        "normalized_text": normalized,
        "tokens": tokens,
        "clean_tokens": clean_tokens,
        "cleaned_text": cleaned_text,
        "pos_tags": pos_tags,
        "entities": entities
    }

text =  "A wonderful not little production. <br /><br />The"
result = preprocess_pipeline(text)
result["cleaned_text"]


# In[ ]:




