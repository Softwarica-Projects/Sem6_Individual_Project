from flask import Flask, render_template, request, jsonify
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from predict.predict import Word2VecReviewAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my-secret-key'

analyzer = Word2VecReviewAnalyzer()

RATINGS_FILE = os.path.join('data', 'review_ratings.json')

def load_review_ratings():
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_review_ratings(ratings):
    os.makedirs('data', exist_ok=True)
    with open(RATINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(ratings, f, indent=2, ensure_ascii=False)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/get-history', methods=['GET'])
def get_history():
    try:
        ratings = load_review_ratings()
        return jsonify({
            'success': True,
            'reviews': ratings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        review_text = data.get('review', '').strip()
        
        if not review_text:
            return jsonify({
                'success': False,
                'error': 'Please enter a review to analyze.'
            }), 400
        
        aspect_scores = analyzer.analyze_review(review_text)
        positive_count = sum(1 for score in aspect_scores.values() if score > 0)
        negative_count = sum(1 for score in aspect_scores.values() if score < 0)
        
        overall_score = sum(aspect_scores.values())
        if positive_count > negative_count:
            overall_sentiment = 'Positive'
        elif negative_count > positive_count:
            overall_sentiment = 'Negative'
        else:
            overall_sentiment = 'Positive' if overall_score > 0 else 'Negative' if overall_score < 0 else 'Neutral'
        
        aspect_data = []
        for aspect, score in aspect_scores.items():
            if aspect != 'miscellaneous' or score != 0:
                sentiment = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
                aspect_data.append({
                    'name': aspect.capitalize(),
                    'score': score,
                    'sentiment': sentiment,
                    'emoji': '😊' if score > 0 else '😞' if score < 0 else '😐'
                })
        
        ratings = load_review_ratings()
        
        new_rating = {
            'timestamp': datetime.now().isoformat(),
            'review': review_text,
            'overall_sentiment': overall_sentiment,
            'overall_score': overall_score,
            'aspect_scores': aspect_scores
        }
        
        ratings.append(new_rating)
        save_review_ratings(ratings)
        
        return jsonify({
            'success': True,
            'review': review_text,
            'overall_sentiment': overall_sentiment,
            'overall_score': overall_score,
            'aspects': aspect_data,
            'method': 'Word2Vec ML',
            'history_count': len(ratings)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        save_review_ratings([])
        return jsonify({
            'success': True,
            'message': 'All reviews cleared successfully.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get-stats', methods=['GET'])
def get_stats():
    ratings = load_review_ratings()
    
    if not ratings:
        return jsonify({
            'success': True,
            'total_reviews': 0,
            'sentiment_distribution': {'Positive': 0, 'Negative': 0},
            'aspect_chart_data': {
                'food': {'pos_count': 0, 'neg_count': 0},
                'service': {'pos_count': 0, 'neg_count': 0},
                'ambience': {'pos_count': 0, 'neg_count': 0},
                'price': {'pos_count': 0, 'neg_count': 0}
            },
            'most_positive_aspect': None,
            'most_negative_aspect': None
        })
    
    total_positive_mentions = 0
    total_negative_mentions = 0
    aspect_totals = {'food': 0, 'service': 0, 'ambience': 0, 'price': 0, 'miscellaneous': 0}
    aspect_counts = {'food': 0, 'service': 0, 'ambience': 0, 'price': 0, 'miscellaneous': 0}
    aspect_sentiment_counts = {
        'food': {'positive': 0, 'negative': 0},
        'service': {'positive': 0, 'negative': 0},
        'ambience': {'positive': 0, 'negative': 0},
        'price': {'positive': 0, 'negative': 0},
        'miscellaneous': {'positive': 0, 'negative': 0}
    }
    
    for entry in ratings:
        for aspect, score in entry['aspect_scores'].items():
            if score != 0:
                aspect_totals[aspect] += score
                aspect_counts[aspect] += 1
                
                if score > 0:
                    aspect_sentiment_counts[aspect]['positive'] += 1
                    total_positive_mentions += 1
                elif score < 0:
                    aspect_sentiment_counts[aspect]['negative'] += 1
                    total_negative_mentions += 1
    
    aspect_averages = {}
    for aspect in aspect_totals:
        if aspect_counts[aspect] > 0:
            aspect_averages[aspect] = round(aspect_totals[aspect] / aspect_counts[aspect], 2)
        else:
            aspect_averages[aspect] = 0
    
    aspect_sentiment_counts_filtered = {k: v for k, v in aspect_sentiment_counts.items() if k != 'miscellaneous'}
    
    aspect_chart_data = {}
    for aspect, counts in aspect_sentiment_counts_filtered.items():
        aspect_chart_data[aspect] = {
            'pos_count': counts['positive'],
            'neg_count': counts['negative']
        }
    
    most_positive_aspect = None
    most_positive_count = 0
    most_negative_aspect = None
    most_negative_count = 0
    
    for aspect, counts in aspect_sentiment_counts_filtered.items():
        if counts['positive'] > most_positive_count:
            most_positive_count = counts['positive']
            most_positive_aspect = aspect
        if counts['negative'] > most_negative_count:
            most_negative_count = counts['negative']
            most_negative_aspect = aspect
    
    return jsonify({
        'success': True,
        'total_reviews': len(ratings),
        'sentiment_distribution': {'Positive': total_positive_mentions, 'Negative': total_negative_mentions},
        'aspect_chart_data': aspect_chart_data,
        'most_positive_aspect': most_positive_aspect,
        'most_negative_aspect': most_negative_aspect,
        'recent_reviews': ratings[-10:]
    })


@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        if not reviews:
            return jsonify({
                'success': False,
                'error': 'No reviews provided.'
            }), 400
        
        results = []
        for review in reviews:
            if review.strip():
                aspect_scores = analyzer.analyze_review(review)
                overall_score = sum(aspect_scores.values())
                overall_sentiment = 'Positive' if overall_score > 0 else 'Negative' if overall_score < 0 else 'Neutral'
                
                results.append({
                    'review': review,
                    'overall_sentiment': overall_sentiment,
                    'aspect_scores': aspect_scores
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)