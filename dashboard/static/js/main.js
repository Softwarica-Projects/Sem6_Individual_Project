const reviewInput = document.getElementById('reviewInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const generateBtn = document.getElementById('generateBtn');
const viewHistoryBtn = document.getElementById('viewHistoryBtn');
const exampleBtn = document.getElementById('exampleBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const overallResult = document.getElementById('overallResult');
const overallScore = document.getElementById('overallScore');
const aspectResults = document.getElementById('aspectResults');
const analyzedReview = document.getElementById('analyzedReview');
const reviews=["the staff was so horrible to us",
    "Unfortunately, my experience here was terrible. The service was slow and disorganized, the food was of very poor quality",
    "Very rude staff, threatening attitude of their staff and thugs.",
    "Service was excellent and waiters were very attentive.",
    "One of the best vibing places I have ever visited.One of the sweetest staff very friendly and made us feel very comfortable.",
"The menu prices are a bit expensive for what you get in quality and portion size.",
"The food was equally impressive. Compliments to Chef Hussain for the delicious and beautifully presented dishes.",
"Overall a very nice experience, incredible food and we would love to come back again.",
"Im not recommend this restaurant they are very roud chefs, was the worst food ever and bad experience",
"Food was excellent but service was very poor",
"Very rude security staff who were acting like they own this place.",
"The service here, the atmosphere and the customer service was fantastic.",
"Unfortunately, my experience here was terrible. The service was slow and disorganized, the food was of very poor quality",
"Foods gone downhill ever since they changed their menu. Used to be really good Turkish grill now it’s bland and no flavour",
"Only good thing about it was the interior. The food was very bad, overpriced and small portions",
"I went there with my wife and I mustbadmist I was surprisung disappointed. We went to this place based on a recommendation and we seriously regret it. The food was dry and cold and simply tasteless. the service was so so bad.",
"Price is too expensive"];
analyzeBtn.addEventListener('click', analyzeReview);
clearBtn.addEventListener('click', clearForm);
generateBtn.addEventListener('click', generateRandomReview);
viewHistoryBtn.addEventListener('click', () => {
    window.location.href = '/history';
});

reviewInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeReview();
    }
});

async function analyzeReview() {
    const review = reviewInput.value.trim();
    
    if (!review) {
        showError('Please enter a review to analyze.');
        return;
    }
    
    hideError();
    resultsSection.style.display = 'none';
    loadingSpinner.style.display = 'block';
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ review: review })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            
            const statsData = await fetch('/get-stats').then(r => r.json());
            loadStats();
            if (statsData.total_reviews > 0) {
                updateTopAspects(statsData.most_positive_aspect, statsData.most_negative_aspect);
            }
        } else {
            showError(data.error || 'An error occurred while analyzing the review.');
        }
    } catch (error) {
        showError('Failed to connect to the server. Please try again.');
        console.error('Error:', error);
    } finally {
        loadingSpinner.style.display = 'none';
    }
}

function displayResults(data) {
    aspectResults.innerHTML = '';
    
    if (data.aspects && data.aspects.length > 0) {
        data.aspects.forEach(aspect => {
            const aspectItem = createAspectItem(aspect);
            aspectResults.appendChild(aspectItem);
        });
    } else {
        aspectResults.innerHTML = '<p style="text-align: center; color: #666;">No specific aspects detected in this review.</p>';
    }
    
    analyzedReview.textContent = `"${data.review}"`;
    
    resultsSection.style.display = 'grid';
    resultsSection.style.opacity = '0';
    setTimeout(() => {
        resultsSection.style.transition = 'opacity 0.5s ease';
        resultsSection.style.opacity = '1';
    }, 10);
}

function createAspectItem(aspect) {
    const div = document.createElement('div');
    div.className = `aspect-item ${aspect.sentiment}`;
    
    const scoreSign = aspect.score > 0 ? '+' : '';
    
    div.innerHTML = `
        <div class="aspect-header">
            <span class="aspect-name">${aspect.name}</span>
            <span class="aspect-emoji">${aspect.emoji}</span>
        </div>
        <div class="aspect-score ${aspect.sentiment}">${scoreSign}${aspect.score}</div>
        <div class="aspect-sentiment">${aspect.sentiment}</div>
    `;
    
    return div;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorMessage.style.display = 'none';
}

function generateRandomReview() {
    const randomIndex = Math.floor(Math.random() * reviews.length);
    const randomReview = reviews[randomIndex];
    reviewInput.value = randomReview;
    reviewInput.focus();
}

function clearForm() {
    reviewInput.value = '';
    resultsSection.style.display = 'none';
    hideError();
    reviewInput.focus();
}

let sentimentChart = null;
let aspectChart = null;

async function loadStats() {
    try {
        const response = await fetch('/get-stats');
        const data = await response.json();
        
        if (data.total_reviews > 0) {
            document.getElementById('totalReviews').textContent = data.total_reviews;
            document.getElementById('positiveCount').textContent = data.sentiment_distribution.Positive || 0;
            document.getElementById('negativeCount').textContent = data.sentiment_distribution.Negative || 0;
            
            renderCharts(data);
            statsSection.style.display = 'block';
            
            return data;
        } else {
            statsSection.style.display = 'none';
            return null;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        return null;
    }
}

function updateTopAspects(mostPositive, mostNegative) {
    const topPositiveEl = document.getElementById('topPositiveAspect');
    const topNegativeEl = document.getElementById('topNegativeAspect');
    
    if (topPositiveEl && mostPositive) {
        topPositiveEl.textContent = mostPositive.toUpperCase();
    } else if (topPositiveEl) {
        topPositiveEl.textContent = 'N/A';
    }
    
    if (topNegativeEl && mostNegative) {
        topNegativeEl.textContent = mostNegative.toUpperCase();
    } else if (topNegativeEl) {
        topNegativeEl.textContent = 'N/A';
    }
}

function renderPieChart(sentimentData) {
    const ctx = document.getElementById('sentimentPieChart');
    if (!ctx) return;
    
    if (sentimentChart) {
        sentimentChart.destroy();
    }
    
    sentimentChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Positive', 'Negative'],
            datasets: [{
                data: [
                    sentimentData.Positive || 0,
                    sentimentData.Negative || 0
                ],
                backgroundColor: [
                    '#10b981',
                    '#ef4444'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

function renderBarChart(aspectChartData) {
    const ctx = document.getElementById('aspectBarChart');
    if (!ctx) return;
    
    if (aspectChart) {
        aspectChart.destroy();
    }
    
    const aspects = Object.keys(aspectChartData);
    const positiveScores = aspects.map(aspect => aspectChartData[aspect].pos_count);
    const negativeScores = aspects.map(aspect => aspectChartData[aspect].neg_count);
    
    aspectChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: aspects.map(a => a.charAt(0).toUpperCase() + a.slice(1)),
            datasets: [
                {
                    label: 'Positive',
                    data: positiveScores,
                    backgroundColor: '#10b981',
                    borderColor: '#059669',
                    borderWidth: 2,
                    barThickness: 30,
                    maxBarThickness: 40
                },
                {
                    label: 'Negative',
                    data: negativeScores,
                    backgroundColor: '#ef4444',
                    borderColor: '#dc2626',
                    borderWidth: 2,
                    barThickness: 30,
                    maxBarThickness: 40
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'x',
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            return Number.isInteger(value) ? value : '';
                        }
                    },
                    grid: {
                        color: '#e5e7eb'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        padding: 10,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            const datasetLabel = context.dataset.label;
                            return `${datasetLabel}: ${value} review${value !== 1 ? 's' : ''}`;
                        }
                    }
                }
            }
        }
    });
}

function renderCharts(data) {
    if (data.sentiment_distribution) {
        renderPieChart(data.sentiment_distribution);
    }
    
    if (data.aspect_chart_data && Object.keys(data.aspect_chart_data).length > 0) {
        renderBarChart(data.aspect_chart_data);
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all review history? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch('/clear-history', {
            method: 'POST'
        });
        
        if (response.ok) {
            statsSection.style.display = 'none';
            
            if (sentimentChart) {
                sentimentChart.destroy();
                sentimentChart = null;
            }
            if (aspectChart) {
                aspectChart.destroy();
                aspectChart = null;
            }
            
            await loadStats();
            
            alert('Review history cleared successfully!');
        } else {
            showError('Failed to clear history. Please try again.');
        }
    } catch (error) {
        showError('Failed to connect to the server. Please try again.');
        console.error('Error:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    reviewInput.focus();
    
    loadStats();
    
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearHistory);
    }
});
