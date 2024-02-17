import requests

API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
pt = query({
	"inputs": "One of the best coffee place in juhu area, staff is very friendly and polite especially Sabina with smiley face always and Aftab who make the best classic frappe. Must try .",
})
def get_sentiment(scores):
    # Initialize variables to store scores
    star_1_score = None
    star_2_score = None
    star_3_score = None
    star_4_score = None
    star_5_score = None
    
    # Extracting scores for each label
    for item in scores[0]:
        if item['label'] == '1 star':
            star_1_score = item['score']
        elif item['label'] == '2 stars':
            star_2_score = item['score']
        elif item['label'] == '3 stars':
            star_3_score = item['score']
        elif item['label'] == '4 stars':
            star_4_score = item['score']
        elif item['label'] == '5 stars':
            star_5_score = item['score']

    # Checking conditions and assigning sentiment
    if (star_1_score and star_1_score > 0.5) or(star_2_score and star_2_score > 0.5) :
        sentiment = "Negative"
    elif star_3_score and star_3_score >= 0.5:
        sentiment = "Neutral"
    elif (star_4_score and star_4_score > 0.5) or (star_5_score and star_5_score > 0.5):
        sentiment = "Positive"
    else:
        sentiment = "Neutral"  # In case none of the conditions match

    print(sentiment)
    return sentiment
get_sentiment(pt)
