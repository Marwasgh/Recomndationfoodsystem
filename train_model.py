import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load preprocessed data
train_data = pd.read_csv('preprocessed_data/train_data.csv')

# Check for any NaN values in 'ingredients' column and replace with empty strings
train_data['ingredients'].fillna('', inplace=True)

# Vectorize the ingredients
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(train_data['ingredients'])

# Save the vectorizer and vectorized data for later use
os.makedirs('models', exist_ok=True)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('models/X_train_vec.pkl', 'wb') as f:
    pickle.dump(X_train_vec, f)

print("Model training complete and saved to models/")
