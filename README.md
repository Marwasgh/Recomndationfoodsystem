# PantryPal: Ingredient-Based Recipe Recommendation System

## Overview

**PantryPal** is a web-based application that helps users discover recipes based on the ingredients they have on hand. By simply entering a list of ingredients, users can get personalized recipe recommendations along with step-by-step cooking instructions and images.

## Features

- **Ingredient-Based Recipe Search**: Input ingredients and get recipe suggestions.
- **Detailed Instructions**: Step-by-step cooking instructions for each recipe.
- **Recipe Images**: Visual representation of the final dish for easy reference.
- **User-Friendly Interface**: Simple and intuitive interface built with Streamlit.

## Project Workflow

### 1. Project Setup

- **Directory Structure**: Organize the project into directories for data, models, and the app.
- **Environment Setup**: Create a virtual environment and install dependencies using `requirements.txt`.
- **Version Control**: Initialize a Git repository and push your code to GitHub.

### 2. Data Collection and Preprocessing

- **Dataset**: Download the "Food Ingredients and Recipe Dataset with Images" from Kaggle.
- **Load Data**: Use pandas to load the dataset.
- **Data Cleaning**: Handle missing values and clean the ingredient lists for consistency.

### 3. Model Training

- **Text Vectorization**: Convert ingredient lists into numerical features using TF-IDF vectorizer.
- **Cosine Similarity**: Use cosine similarity to find recipes with similar ingredients.
- **Model Saving**: Save the trained vectorizer and cosine similarity matrix for later use.

### 4. Building the Streamlit App

- **User Interface**: Create a simple UI where users can input ingredients in a text box.
- **Recipe Recommendation**: Load the model and data, process the user input, and display recommended recipes.
- **Display Instructions and Images**: Show step-by-step instructions and corresponding images for each recommended recipe.

### 5. Deployment

- **Local Testing**: Test the Streamlit app locally to ensure it works correctly.
- **Deploy to Streamlit**: Deploy the app on Streamlit Sharing or another cloud platform.

## Installation

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/PantryPal.git
    cd PantryPal
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

## Usage

1. Open the app in your browser.
2. Enter the ingredients you have in the text box (separated by commas).
3. Click on "Get Recipes".
4. View the recommended recipes along with instructions and images.

## Dataset

The dataset used for this project is the "Food Ingredients and Recipe Dataset with Images" available on Kaggle. It includes a collection of recipes with ingredients, instructions, and images.

## File Structure

- `data/`: Directory containing the dataset.
- `models/`: Directory for saving trained models.
- `app.py`: Streamlit app script.
- `data_preprocessing.py`: Script for data cleaning and preprocessing.
- `train_model.py`: Script for training the model.
- `requirements.txt`: List of dependencies.

## Contributing

Contributions are welcome! If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Kaggle for providing the dataset.
- Streamlit for the web application framework.
