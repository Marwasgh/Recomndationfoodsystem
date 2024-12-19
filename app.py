import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load preprocessed data and model components
train_data = pd.read_csv('preprocessed_data/train_data.csv')

# Check if 'Image_Name' column exists
if 'Image_Name' not in train_data.columns:
    st.error("The 'Image_Name' column is missing from the dataset.")
else:
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/X_train_vec.pkl', 'rb') as f:
        X_train_vec = pickle.load(f)

    # Function to recommend recipes based on ingredients
    def recommend_recipes(ingredients):
        ingredients_vec = vectorizer.transform([' '.join(ingredients)])
        similarities = cosine_similarity(ingredients_vec, X_train_vec).flatten()
        
        # Appliquer un seuil ou vérifier les meilleures similarités
        threshold = 0.1  # Exclure les résultats avec une similarité trop faible
        indices = [i for i, sim in enumerate(similarities) if sim > threshold]
        
        # Vérifier si des correspondances existent
        if not indices:
            return None  # Auc
        else:
            indices = sorted(indices, key=lambda i: similarities[i], reverse=True)[:3]  # Obtenir les 3 meilleurs
            return train_data.iloc[indices][['Title', 'Instructions', 'Image_Name']]
     
    # Function to format instructions
    def format_instructions(instructions):
        steps = instructions.split('. ')
        formatted_steps = [f"**Step {i+1}:** {step.strip()}." for i, step in enumerate(steps) if step]
        formatted_steps.append("**You're ready to feast!**")
        return '\n\n'.join(formatted_steps)

    # Streamlit app
    st.markdown("<h1 style='color:Green;'>Food Recommendation System</h1>", unsafe_allow_html=True)

    st.header('Enter the ingredients you have:')
    ingredients_input = st.text_area("Enter ingredients separated by commas", "")

    if st.button('Get Recipes'):
        if ingredients_input:
            ingredients_list = [ingredient.strip().lower() for ingredient in ingredients_input.split(',')]
            recommended_recipes = recommend_recipes(ingredients_list)
            
            if recommended_recipes is None or recommended_recipes.empty:
                st.error("No recipes found for the given ingredients. Please try again with different ingredients!")
            else:
                st.write("Recommended Recipes:")
                for index, row in recommended_recipes.iterrows():
                    st.write(f"### {row['Title']}")
                    st.markdown(format_instructions(row['Instructions']))
                    image_path = f"dataset/FoodImages/FoodImages/{row['Image_Name']}"
                    if not image_path.endswith(('.jpg', '.png', '.jpeg')):
                        image_path += '.jpg'  
                        st.write(f"Image Path: {image_path}")
                    if os.path.exists(image_path):
                        st.image(image_path, use_column_width=True)
                    else:
                     st.write("Image not available ", {image_path})
        else:
            st.warning("Please enter some ingredients.")