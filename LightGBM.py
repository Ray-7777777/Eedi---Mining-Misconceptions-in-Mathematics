import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')
# Load the data
train_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
misconception_map = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
sample_sub = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv')

# Preprocessing
def preprocess_data(df):
    df['full_text'] = df['QuestionText'] + " " + \
                     df['AnswerAText'] + " " + \
                     df['AnswerBText'] + " " + \
                     df['AnswerCText'] + " " + \
                     df['AnswerDText']
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Prepare distractor data
def prepare_distractor_data(df, answer_option, is_train=True):
    distractor_df = df.copy()
    distractor_df['AnswerText'] = distractor_df[f'Answer{answer_option}Text']
    distractor_df['QuestionId_Answer'] = distractor_df['QuestionId'].astype(str) + "_" + answer_option
    
    if is_train:
        distractor_df['MisconceptionId'] = distractor_df[f'Misconception{answer_option}Id']
        if answer_option in ['A', 'B', 'C', 'D']:
            distractor_df = distractor_df[distractor_df['CorrectAnswer'] != answer_option]
    
    return distractor_df

# Prepare training data
train_distractors = []
for option in ['A', 'B', 'C', 'D']:
    train_distractors.append(prepare_distractor_data(train_df, option, is_train=True))
train_data = pd.concat(train_distractors).dropna(subset=['MisconceptionId'])

# Prepare test data
test_distractors = []
for option in ['A', 'B', 'C', 'D']:
    test_distractors.append(prepare_distractor_data(test_df, option, is_train=False))
test_data = pd.concat(test_distractors)

# Feature engineering
def create_features(df):
    df['QuestionLength'] = df['QuestionText'].apply(len)
    df['AnswerLength'] = df['AnswerText'].apply(len)
    df['ConstructId'] = df['ConstructId'].astype('category')
    df['SubjectId'] = df['SubjectId'].astype('category')
    return df

train_data = create_features(train_data)
test_data = create_features(test_data)

# Text vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
full_text_tfidf = tfidf.fit_transform(train_data['full_text'])
answer_text_tfidf = tfidf.transform(train_data['AnswerText'])

# Combine features
X = hstack([
    full_text_tfidf,
    answer_text_tfidf,
    train_data[['QuestionLength', 'AnswerLength']].values
])

# Target encoding
le = LabelEncoder()
y = le.fit_transform(train_data['MisconceptionId'])

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

train_data_lgb = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

model = lgb.train(
    params,
    train_data_lgb,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ]
)

# Prepare test features
test_full_text_tfidf = tfidf.transform(test_data['full_text'])
test_answer_text_tfidf = tfidf.transform(test_data['AnswerText'])

X_test = hstack([
    test_full_text_tfidf,
    test_answer_text_tfidf,
    test_data[['QuestionLength', 'AnswerLength']].values
])

# Predict and create submission
y_probs = model.predict(X_test, num_iteration=model.best_iteration)
top_k = 25
top_classes = np.argsort(-y_probs, axis=1)[:, :top_k]

# Convert to integers and format as strings without .0
predicted_misconceptions = le.inverse_transform(top_classes.flatten()).reshape(top_classes.shape)
predicted_misconceptions = predicted_misconceptions.astype(int)

# Create submission dataframe
submission = pd.DataFrame({
    'QuestionId_Answer': test_data['QuestionId_Answer'],
    'MisconceptionId': [' '.join(map(str, row)) for row in predicted_misconceptions]
})

# Ensure we have all required rows in correct order
submission = submission.set_index('QuestionId_Answer').loc[sample_sub['QuestionId_Answer']].reset_index()

# Save to CSV without index and with correct header
submission.to_csv('submission.csv', index=False)
print("Submission file created with proper format!")

# Display first few rows to verify format
print("\nSample submission output:")
print(submission.head())