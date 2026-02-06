import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from gensim.models import KeyedVectors
import spacy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_parser import SemEvalDataParser
from config.constants import WORD2VEC_PATH, VECTOR_SIZE

nlp = spacy.load("en_core_web_sm")


class Word2VecAspectSentimentModel:
    def __init__(self, word2vec_path=None):
        self.word2vec_path = word2vec_path or WORD2VEC_PATH
        self.word2vec_model = None
        self.vector_size = VECTOR_SIZE

        self.aspect_mlb = MultiLabelBinarizer()
        self.aspect_classifier = None

        self.sentiment_classifier = None

        self.aspect_categories = ['food', 'service', 'ambience', 'price']

    def load_word2vec(self):
        if not os.path.exists(self.word2vec_path):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"Word2Vec model not found"
                f"{'='*60}\n"
            )
        print(f"Loading Word2Vec model")
        self.word2vec_model = KeyedVectors.load_word2vec_format(
            self.word2vec_path,
            binary=True,
            limit=500000
        )

    def sentence_to_vector(self, text):
        if self.word2vec_model is None:
            self.load_word2vec()

        doc = nlp(text.lower())
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]

        vectors = []
        for word in tokens:
            if word in self.word2vec_model:
                vectors.append(self.word2vec_model[word])

        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)

    def texts_to_vectors(self, texts):
        return np.array([self.sentence_to_vector(text) for text in texts])

    def train(self, aspect_data, sentiment_data):
        print("\n" + "="*60)
        print("TRAINING WORD2VEC-BASED MODELS")
        print("="*60)

        self.load_word2vec()

        print("\nTraining Aspect Classification Model...")
        self._train_aspect_model(aspect_data)

        print("\nTraining Sentiment Classification Model...")
        self._train_sentiment_model(sentiment_data)

        print("\nSaving models...")
        self._save_models()

    def _train_aspect_model(self, aspect_data):
        sentences = [item[0] for item in aspect_data]
        aspects = [item[1] for item in aspect_data]

        y = self.aspect_mlb.fit_transform(aspects)
        X = self.texts_to_vectors(sentences)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        base_classifier = LinearSVC(random_state=42, max_iter=2000)

        from sklearn.multioutput import MultiOutputClassifier
        self.aspect_classifier = MultiOutputClassifier(base_classifier)

        self.aspect_classifier.fit(X_train, y_train)

        y_pred = self.aspect_classifier.predict(X_test)

        exact_match = accuracy_score(y_test, y_pred)
        hamming = 1 - hamming_loss(y_test, y_pred)

        print(f"Exact Match Accuracy: {exact_match*100:.2f}%")
        print(f"Hamming Score (per-label): {hamming*100:.2f}%")

    def _train_sentiment_model(self, sentiment_data):
        sentences = [item[0] for item in sentiment_data]
        sentiments = [item[2] for item in sentiment_data]

        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        y = np.array([sentiment_map.get(s, 0) for s in sentiments])
        X = self.texts_to_vectors(sentences)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.sentiment_classifier = LinearSVC(random_state=42, max_iter=2000)

        self.sentiment_classifier.fit(X_train, y_train)

        y_pred = self.sentiment_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("\n")
        print(f"Sentiment Classification Results:")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")

        cv_scores = cross_val_score(self.sentiment_classifier, X_train, y_train, cv=5)
        print(f"Mean Cross Validation Accuracy: {cv_scores.mean()*100:.2f}% (Standard Dev: {cv_scores.std()*100:.2f}%)")

    def _save_models(self):
        os.makedirs('models', exist_ok=True)
        with open('models/w2v_aspect_mlb.pkl', 'wb') as f:
            pickle.dump(self.aspect_mlb, f)
        with open('models/w2v_aspect_classifier.pkl', 'wb') as f:
            pickle.dump(self.aspect_classifier, f)
        with open('models/w2v_sentiment_classifier.pkl', 'wb') as f:
            pickle.dump(self.sentiment_classifier, f)


def load_semeval_data():
    print("Loading SemEval 2014 data")
    parser = SemEvalDataParser('data/Restaurants_Train.xml')
    sentences, labels = parser.parse()

    aspect_data = []
    for sent, label_vector in zip(sentences, labels):
        aspects = [parser.aspect_labels[i] for i in range(len(label_vector)) if label_vector[i] == 1]
        aspect_data.append((sent, aspects))
    sentiment_dicts = parser.parse_with_sentiment()
    sentiment_data = []
    for item in sentiment_dicts:
        text = item['text']
        for aspect, score in item['aspect_sentiments'].items():
            if score != 0:
                polarity = 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')
                sentiment_data.append((text, aspect, polarity))
   
    return aspect_data, sentiment_data


if __name__ == "__main__":
    aspect_data, sentiment_data = load_semeval_data()
    model = Word2VecAspectSentimentModel()
    
    try:
        model.train(aspect_data, sentiment_data)
        print("\n" + "="*60)
        print("Model trained and saved to models folder.")
        print("="*60)
        
    except FileNotFoundError as e:
        print(str(e))
