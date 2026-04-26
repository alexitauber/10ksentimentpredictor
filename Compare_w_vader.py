import pandas as pd
import re

def calculate_sentiment_score(text, dictionary_path):
    # 1. Load the dictionary CSV
    # Ensure that it reads the columns 'Word', 'Neg', and 'Pos'
    df = pd.read_csv(dictionary_path)
    
    # Drop any missing words just in case
    df = df.dropna(subset=['Word'])
    
    # 2. Create a fast lookup dictionary
    # Assuming 'Pos' contains +1 and 'Neg' contains -1 (or 0)
    # We combine them so a positive word is 1, negative is -1, neutral is 0.
    # Convert 'Word' to string to prevent errors with anomalous data types
    sentiment_dict = pd.Series(
        df['Pos'].values + df['Neg'].values, 
        index=df['Word'].astype(str).str.upper()
    ).to_dict()
    
    # 3. Tokenize the sample text
    # Convert the text to uppercase to match the dictionary format
    # re.findall(r'\b[A-Z]+\b', ...) extracts only alphabetic words
    words = re.findall(r'\b[A-Z]+\b', text.upper())
    
    # 4. Calculate the score
    total_score = 0
    positive_words_found = []
    negative_words_found = []
    
    for word in words:
        if word in sentiment_dict:
            score = sentiment_dict[word]
            total_score += score
            
            # Keep track of which words triggered the scores
            if score > 0:
                positive_words_found.append(word)
            elif score < 0:
                negative_words_found.append(word)
                
    return {
        'total_score': total_score,
        'positive_words': positive_words_found,
        'negative_words': negative_words_found
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Put your sample text here
    sample_text = """
    The company reported excellent and outstanding profits this quarter. 
    However, they abandoned the failing project due to a severe deficit and bad management.
    """
    
    # Make sure the CSV file is in the same directory as this script
    csv_filename = 'LM_MasterDictionary_1993-2021.csv'
    
    results = calculate_sentiment_score(sample_text, csv_filename)
    
    print(f"Overall Sentiment Score: {results['total_score']}")
    print(f"Positive Words Found: {results['positive_words']}")
    print(f"Negative Words Found: {results['negative_words']}")