import xml.etree.ElementTree as ET
import os
from typing import List, Dict, Tuple


class SemEvalDataParser:
    CATEGORY_MAPPING = {
        'food': 'food',
        'service': 'service',
        'ambience': 'ambience',
        'price': 'price',
        'miscellaneous': 'miscellaneous'
    }
    
    POLARITY_MAPPING = {
        'positive': 1,
        'negative': -1,
        'neutral': 0,
        'conflict': 0
    }
    
    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.sentences = []
        self.aspect_labels = ['food', 'service', 'ambience', 'price', 'miscellaneous']
    
    def parse(self) -> Tuple[List[str], List[List[int]]]:
        if not os.path.exists(self.xml_file_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")
        
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        
        sentences = []
        labels = []
        
        for sentence in root.findall('sentence'):
            text_elem = sentence.find('text')
            if text_elem is None or not text_elem.text:
                continue
            
            text = text_elem.text.strip()
            
            aspect_categories = sentence.find('aspectCategories')
            if aspect_categories is None:
                continue
            
            label_vector = [0, 0, 0, 0, 0]
            
            for aspect_cat in aspect_categories.findall('aspectCategory'):
                category = aspect_cat.get('category', '').lower()
                
                mapped_category = self.CATEGORY_MAPPING.get(category)
                
                if mapped_category and mapped_category in self.aspect_labels:
                    idx = self.aspect_labels.index(mapped_category)
                    label_vector[idx] = 1
            
            if sum(label_vector) > 0:
                sentences.append(text)
                labels.append(label_vector)
        
        return sentences, labels
    
    def parse_with_sentiment(self) -> List[Dict]:
        if not os.path.exists(self.xml_file_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")
        
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        
        results = []
        
        for sentence in root.findall('sentence'):
            text_elem = sentence.find('text')
            if text_elem is None or not text_elem.text:
                continue
            
            text = text_elem.text.strip()
            
            aspect_categories = sentence.find('aspectCategories')
            if aspect_categories is None:
                continue
            
            aspect_sentiments = {
                'food': 0,
                'service': 0,
                'ambience': 0,
                'price': 0,
                'miscellaneous': 0
            }
            
            for aspect_cat in aspect_categories.findall('aspectCategory'):
                category = aspect_cat.get('category', '').lower()
                polarity = aspect_cat.get('polarity', '').lower()
                
                mapped_category = self.CATEGORY_MAPPING.get(category)
                sentiment_score = self.POLARITY_MAPPING.get(polarity, 0)
                
                if mapped_category:
                    aspect_sentiments[mapped_category] += sentiment_score
            
            results.append({
                'text': text,
                'aspect_sentiments': aspect_sentiments
            })
        
        return results
    
    def get_statistics(self) -> Dict:
        sentences, labels = self.parse()
        
        stats = {
            'total_sentences': len(sentences),
            'aspect_counts': {},
            'multi_aspect_sentences': 0
        }
        
        for i, aspect in enumerate(self.aspect_labels):
            count = sum(1 for label in labels if label[i] == 1)
            stats['aspect_counts'][aspect] = count
        
        stats['multi_aspect_sentences'] = sum(1 for label in labels if sum(label) > 1)
        
        return stats


def test_parser():
    data_path = os.path.join('data', 'Restaurants_Train.xml')
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    
    parser = SemEvalDataParser(data_path)
    
    sentences, labels = parser.parse()
    
    print(f"\nSample sentences:")
    for i in range(min(5, len(sentences))):
        print(f"\n{i+1}. {sentences[i]}")
        aspects = [parser.aspect_labels[j] for j in range(len(labels[i])) if labels[i][j] == 1]
        print(f"   Aspects: {', '.join(aspects)}")
    
    stats = parser.get_statistics()
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Multi-aspect sentences: {stats['multi_aspect_sentences']}")


if __name__ == "__main__":
    test_parser()
