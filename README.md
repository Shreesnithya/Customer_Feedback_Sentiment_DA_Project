# Customer Feedback Sentiment Analysis

**Intern ID:** CITS8439 | **Author:** Nithyashree S

Sentiment analysis of customer reviews: data cleaning, exploratory analysis, text features (TF-IDF) and a Logistic Regression model that classifies feedback as Positive, Neutral or Negative.

## Goal
Find out how customers feel about the products, which words signal each sentiment, and how well a model can classify new feedback.

## Dataset
- Women's E-Commerce Clothing Reviews (Kaggle, by nicapotato): 23,486 reviews × 11 columns
- Main columns: Title, Review Text, Rating (1 to 5), Recommended IND, Department Name, Class Name
- Sentiment label created from the rating: 1-2 = Negative, 3 = Neutral, 4-5 = Positive

## Tools
Python: pandas, matplotlib, seaborn, scikit-learn

## Data cleaning
- 0 duplicate rows
- Dropped the extra row-number column
- Removed 845 reviews with no review text (22,641 reviews remain)
- Filled 3,810 missing titles with a blank and joined title and review into one text field
- The rating was used only to create the label, never as a model input (to avoid leakage)

## Exploration
- Positive 77.1% (17,448), Neutral 12.5% (2,823), Negative 10.5% (2,370): the classes are imbalanced.
- Review length is similar across sentiments (about 60 words on average), so length does not separate them.

![Reviews by sentiment](Customer%20feedback%20sentiment%20project/sentiment_distribution.png)

## Method
1. Split the reviews 80/20 (18,112 train / 4,529 test), stratified by sentiment
2. Converted the text to TF-IDF features (single words and word pairs, 20,000 features), fitted on the training set only
3. Trained Logistic Regression with balanced class weights
4. Evaluated with precision, recall and F1 per class and a confusion matrix

## Model results (test set: 4,529 reviews)
Accuracy: **0.815** (always guessing "Positive" would give 0.771). Macro F1: **0.65**.

| Class | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Negative | 0.54 | 0.63 | 0.58 | 474 |
| Neutral | 0.39 | 0.52 | 0.45 | 565 |
| Positive | 0.96 | 0.89 | 0.92 | 3,490 |

![Confusion matrix](Customer%20feedback%20sentiment%20project/confusion_matrix.png)

- Positive reviews are predicted well; Negative moderately; Neutral is the weakest class, because 3-star reviews mix positive and negative language.
- Most mistakes involve Neutral. Only 104 of 4,529 reviews (about 2.3%) were flipped between Positive and Negative.

## Words that signal each sentiment
- **Negative:** disappointed, horrible, unflattering, poor, huge, awful, cheap, terrible, "going back"
- **Positive:** love, perfect, great, comfortable, comfy, beautiful, flattering, soft
- **Neutral:** however, but, unfortunately, "just ok", meh, "returning it"

![Top words by sentiment](Customer%20feedback%20sentiment%20project/top_words.png)

The model also labelled three made-up test reviews as expected (a happy review Positive, a complaint Negative, a mixed "okay, however..." review Neutral).

## Recommendations
1. Improve size guides and fit information, since complaints use words like "huge" and "unflattering".
2. Review fabric quality on the worst-rated items ("poor", "cheap").
3. Watch return language ("going back", "returning it") as a sign of unhappy customers.
4. Use the model to route likely-negative feedback to customer care first.
5. Treat Neutral reviews as useful feedback: customers who like the product but have one specific problem.

## Limitations
- Labels come from star ratings, which are an imperfect proxy for the feeling in the text.
- Neutral (3-star) reviews mix positive and negative language.
- Single train/test split; cross-validation would give a steadier estimate.
- Data covers women's clothing reviews from one platform. The word weights show association, not cause.

## Files
All files are in the `Customer feedback sentiment project` folder:
- `Customer_feedback_sentiment_project.ipynb`: full analysis notebook
- `Womens Clothing E-Commerce Reviews.csv`: dataset
- `sentiment_distribution.png`, `confusion_matrix.png`, `top_words.png`: charts used in this README
- [View the notebook on nbviewer](https://nbviewer.org/github/Shreesnithya/Customer_Feedback_Sentiment_DA_Project/blob/main/Customer%20feedback%20sentiment%20project/Customer_feedback_sentiment_project.ipynb)

## How to run
Download the notebook and the CSV into the same folder, check the file name in the loading cell, and run all cells.


