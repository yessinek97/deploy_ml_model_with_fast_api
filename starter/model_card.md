# Model Card

# Model Details
- Owner: Udacity Trainee
- Date: 10/28/2024
- Version: 1.0.0, this is the initial version of the model
- Model type: Machine learning model, 
random forest classifier imported from the scikit-learn library

# Intended Use
- This model is trained to predict an indiviual's salary based on a set of features,
it outputs discrete (binary) values indicating whether ot not a given individual's salary 
is above or below 50k/year.

# Training Data
- Census dataset, also known as adult dataset, extracted by Barry Becker in 1994. It comprises
48842 entries, each described with 14 features (categorical and continuous).(Note that this data has mssing values).

# Evaluation Data
- Evaluation data is a subset of the training data (20%).

# Metrics
- The metrics considered for the evaluation of this model on the task described in the previous sections:
    * Accuracy: 0.966
    * F1-score: 0.94 (calculated as a fucntion of precision and recall)
    * True Positive Rate (TPR): 0.987
    * True Negative Rate (TNR): 0.931
    * False Positive Rate (FPR) and False Negative Rate (FNR) can be inferred from TPR and TNR

# Ethical Considerations
- This project was implemented as part of a Udacity nanodegree program, and was built upon
a provided starter code, and is not accomplished by the trainee from scratch. This work is within the frame of the training 
and is not intended for production or monetary puposes.

# Caveats and Recommendations
- This model is only evaluated on a subset of the training data, thus no proof of generalization
is provided. It is recommended to test this model on external datasets, to objectively judge the model's general performance.
