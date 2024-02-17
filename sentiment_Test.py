import csv
import requests

API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"

# List of API tokens
api_tokens = [
    "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",
    "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",
    "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",
    "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"
]

# Initialize token index
current_token_index = 0

# Function to rotate API token
def get_next_token():
    global current_token_index
    current_token_index = (current_token_index + 1) % len(api_tokens)
    return api_tokens[current_token_index]

def query(payload):
    api_token = get_next_token()
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_sentiment(scores):
    if isinstance(scores, list) and scores:
        star_1_score = None
        star_2_score = None
        star_3_score = None
        star_4_score = None
        star_5_score = None
        
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

        if (star_1_score and star_1_score > 0.5) or (star_2_score and star_2_score > 0.5):
            sentiment = "Negative"
        elif star_3_score and star_3_score >= 0.5:
            sentiment = "Neutral"
        elif (star_4_score and star_4_score > 0.5) or (star_5_score and star_5_score > 0.5):
            sentiment = "Positive"
        else:
            sentiment = "Neutral"
    else:
        sentiment = "Unknown"
    
    return sentiment

def process_reviews_from_csv(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        fieldnames = csv_reader.fieldnames + ['Sentiment']  # Add a new column for sentiment
        reviews = []
        for row in csv_reader:
            pt = query({"inputs": row['Review']})
            sentiment = get_sentiment(pt)
            row['Sentiment'] = sentiment
            reviews.append(row)

    with open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)

# Example usage
input_csv_filename = 'abcoffee_scrape_data.csv'
output_csv_filename = 'abcoffee_scrape_data2.csv'
process_reviews_from_csv(input_csv_filename, output_csv_filename)
