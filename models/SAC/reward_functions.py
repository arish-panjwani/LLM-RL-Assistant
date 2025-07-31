def clarity_consistency_reward(cosine_similarity, lexical_redundancy=0.0, groq_clarity_score=0.0,
                                lambda1=0.8, lambda2=0.5, lambda3=1.0):
    '''
    Reward for clarity and consistency in prompts
    '''
    return lambda1 * cosine_similarity - lambda2 * lexical_redundancy + lambda3 * groq_clarity_score

def relevance_reward(user_rating, sentiment_score, engagement_length=0,
                     alpha=1.0, beta=0.5, gamma=0.1):
    '''
    Reward for response relevance using user feedback
    '''
    return alpha * user_rating + beta * sentiment_score + gamma * engagement_length

def hallucination_penalty_reward(base_reward, hallucination_score, delta=1.0):
    '''
    Reward with hallucination penalty
    '''
    return base_reward - delta * hallucination_score
