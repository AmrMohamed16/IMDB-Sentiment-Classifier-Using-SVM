# IMDB-Sentiment-Classifier-Using-SVM
Created for the Natural Language Processing [CS411P] course at Mansoura University, this project uses SVM to classify IMDB movie reviews as positive or negative. With advanced NLP techniques, including text preprocessing and TF-IDF vectorization, the model delivers accurate sentiment predictions.


# Table Of Content 

* [Brief](#Brief)
* [DataSet](#DataSet)
* [How It Works](#HowItWorks)
* [Tools](#Tools)
* [Remarks](#Remarks)
* [Usage](#Usage)
* [Sample Run](#SampleRun)

  # Brief
  This project demonstrates how to classify IMDB movie reviews into positive or negative sentiment using Support Vector Machines (SVM). It applies Natural Language Processing (NLP)      techniques like text preprocessing and TF-IDF vectorization to convert text data into features suitable for machine learning models.


   # DataSet
  The dataset used in this project consists of movie reviews from [IMDB](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews), each labeled with a positive or negative         sentiment.
     The data includes text-based reviews that are processed to extract useful features for the classification model.


  # How_It_Works

     * Text Preprocessing: Reviews are cleaned using regular expressions, tokenized, and lemmatized with a Porter Stemmer to remove noise and ensure consistency.
    
     * Vectorization: TF-IDF vectorization is used to convert the text into numerical features, representing the importance of words in each review.
    
     * Model Training: The data is split into training and testing sets, and an SVM classifier is trained to predict the sentiment of the reviews.
    
     * Evaluation: Model performance is evaluated using accuracy, precision, recall, and F1-score metrics.

    
 
  # Tools 

  1. Jupyter Notebook & VS Code

  2. Python 3.x

  3. pandas

  4. re

  5. nltk

  6. scikit-learn

  7. pickle


  # Remarks

    This project showcases how to effectively utilize SVM for text classification tasks, combining NLP techniques with machine learning for high-performance sentiment analysis.
     The use of TF-IDF for feature extraction ensures that the model captures the most meaningful words from the reviews.


  # Usage

  1. Clone the repository:
     
     ```bash
     git clone https://github.com/AmrMohamed16/IMDB-Sentiment-Classifier-Using-SVM

  2. Navigate to the cloned repository:
     
     ```bash
     cd IMDB-Sentiment-Classifier-Using-SVM

  4. Run the Jupyter Notebook:
     
     ```bash
     Sentiment Analysis.ipynb

  5. Run app window:
     ```bash
     streamlit run app.py


  # Sample Run

  * Positive Sentiment
    <img width="1018" alt="Screenshot 2024-11-21 at 8 42 14 AM" src="https://github.com/user-attachments/assets/ac1b6667-18ed-4fc6-91f6-d8b4f6133562">
    <hr>

  * Negative Sentiment
    <img width="1018" alt="Screenshot 2024-11-21 at 8 45 51 AM" src="https://github.com/user-attachments/assets/a82cfeeb-5ee9-4226-8600-b5042081b2b9">
     
     






  




  
