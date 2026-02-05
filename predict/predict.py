import pickle
import numpy as np
import spacy
import re
import os
import sys
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.constants import WORD2VEC_PATH, VECTOR_SIZE

nlp = spacy.load("en_core_web_sm")

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'nothing', 'nowhere', "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't"}
stop_words = stop_words - negation_words

class Word2VecReviewAnalyzer:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.models_dir = models_dir
        
        self.word2vec_path = WORD2VEC_PATH
        self.vector_size = VECTOR_SIZE
        
        print("Loading Word2Vec model...")
        if not os.path.exists(self.word2vec_path):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"Word2Vec model not found"
                f"{'='*60}\n"
            )
        
        self.word2vec_model = KeyedVectors.load_word2vec_format(
            self.word2vec_path,
            binary=True,
            limit=500000
        )        
        with open(os.path.join(models_dir, 'w2v_aspect_mlb.pkl'), 'rb') as f:
            self.aspect_mlb = pickle.load(f)
        with open(os.path.join(models_dir, 'w2v_aspect_classifier.pkl'), 'rb') as f:
            self.aspect_classifier = pickle.load(f)
        
        with open(os.path.join(models_dir, 'w2v_sentiment_classifier.pkl'), 'rb') as f:
            self.sentiment_classifier = pickle.load(f)        
        self.aspect_categories = self.aspect_mlb.classes_
    
    def sentence_to_vector(self, text):
        doc = nlp(text.lower())
        
        tokens = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            word = token.text
            if word not in stop_words:
                tokens.append(word)
        
        vectors = []
        for word in tokens:
            if word in self.word2vec_model:
                vectors.append(self.word2vec_model[word])
        
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def analyze_review(self, review_text):
        aspect_scores = {aspect: 0 for aspect in self.aspect_categories}
        aspect_counts = {aspect: 0 for aspect in self.aspect_categories}
        
        doc = nlp(review_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for sent in sentences:
            sub_sents = re.split(r'\s+but\s+|\s+however\s+|\s+though\s+|\s+although\s+', 
                                sent, flags=re.IGNORECASE)
            
            for sentence in sub_sents:
                sentence = sentence.strip()
                
                if len(sentence.split()) < 2:
                    continue
                
                vec = self.sentence_to_vector(sentence).reshape(1, -1)
                
                aspects_pred = self.aspect_classifier.predict(vec)[0]
                
                for i, aspect in enumerate(self.aspect_categories):
                    if aspects_pred[i] == 1:
                        sentiment_score = self.sentiment_classifier.predict(vec)[0]
                        
                        if sentiment_score != 0:
                            aspect_scores[aspect] += sentiment_score
                            aspect_counts[aspect] += 1
        
        final_scores = {}
        for aspect in self.aspect_categories:
            if aspect_counts[aspect] > 0:
                avg_score = aspect_scores[aspect] / aspect_counts[aspect]
                final_scores[aspect] = int(round(avg_score))
            else:
                final_scores[aspect] = 0
        
        return final_scores
    
    