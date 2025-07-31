from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()

def compute_cosine_similarity(text1: str, text2: str) -> float:
    emb1 = embedding_model.encode([text1])[0]
    emb2 = embedding_model.encode([text2])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])

def compute_sentiment_score(text: str) -> float:
    score = sentiment_analyzer.polarity_scores(text)
    return score['compound']

def normalize_user_rating(rating_raw) -> int:
    if isinstance(rating_raw, str):
        if rating_raw.lower() in ['up', 'thumbs up', 'positive']:
            return 1
        elif rating_raw.lower() in ['down', 'thumbs down', 'negative']:
            return -1
        else:
            return 0
    elif isinstance(rating_raw, int):
        if rating_raw >= 4:
            return 1
        elif rating_raw <= 2:
            return -1
        else:
            return 0
    return 0

def calculate_reward(cos_sim, sentiment, user_rating, alpha=1.0, beta=0.5, lambda1=0.8):
    reward = lambda1 * cos_sim + beta * sentiment + alpha * user_rating
    return reward
