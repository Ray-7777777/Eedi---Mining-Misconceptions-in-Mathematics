import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
sample_sub = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv')

# Preprocessing
def preprocess_data(df):
    df['full_text'] = df['QuestionText'].fillna('') + " " + \
                     df['AnswerAText'].fillna('') + " " + \
                     df['AnswerBText'].fillna('') + " " + \
                     df['AnswerCText'].fillna('') + " " + \
                     df['AnswerDText'].fillna('')
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
    df['QuestionLength'] = df['QuestionText'].apply(lambda x: len(str(x)))
    df['AnswerLength'] = df['AnswerText'].apply(lambda x: len(str(x)))
    df['ConstructId'] = df['ConstructId'].astype('category').cat.codes
    df['SubjectId'] = df['SubjectId'].astype('category').cat.codes
    return df

train_data = create_features(train_data)
test_data = create_features(test_data)

# Text vectorization
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
full_text_tfidf = tfidf.fit_transform(train_data['full_text'])
answer_text_tfidf = tfidf.transform(train_data['AnswerText'])

# Combine features
X = hstack([
    full_text_tfidf,
    answer_text_tfidf,
    train_data[['QuestionLength', 'AnswerLength', 'ConstructId', 'SubjectId']].values
])

# Target encoding
le = LabelEncoder()
y = le.fit_transform(train_data['MisconceptionId'])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert sparse to dense for Logistic Regression
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_val_scaled = scaler.transform(X_val.astype(np.float64))

# Train LightGBM
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'verbose': -1
}

lgb_model = lgb.train(
    lgb_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Train Logistic Regression
ridge = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    solver='lbfgs',
    C=0.1,
    penalty='l2'
)
ridge.fit(X_train_scaled, y_train)

# Prepare test features
test_full_text_tfidf = tfidf.transform(test_data['full_text'])
test_answer_text_tfidf = tfidf.transform(test_data['AnswerText'])

X_test = hstack([
    test_full_text_tfidf,
    test_answer_text_tfidf,
    test_data[['QuestionLength', 'AnswerLength', 'ConstructId', 'SubjectId']].values
])
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# Get predictions
lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
ridge_probs = ridge.predict_proba(X_test_scaled)

# Align probabilities for missing classes
if ridge_probs.shape[1] < len(le.classes_):
    full_probs = np.zeros((ridge_probs.shape[0], len(le.classes_)))
    for i, cls in enumerate(ridge.classes_):
        full_probs[:, cls] = ridge_probs[:, i]
    ridge_probs = full_probs

# Ensemble predictions
ensemble_probs = (lgb_probs + ridge_probs) / 2
top_k = 25
top_classes = np.argsort(-ensemble_probs, axis=1)[:, :top_k]

# Create submission
predicted_misconceptions = le.inverse_transform(top_classes.flatten()).reshape(top_classes.shape)
predicted_misconceptions = predicted_misconceptions.astype(int)

submission = pd.DataFrame({
    'QuestionId_Answer': test_data['QuestionId_Answer'],
    'MisconceptionId': [' '.join(map(str, row)) for row in predicted_misconceptions]
})

# Align with sample submission
submission = submission.set_index('QuestionId_Answer').loc[sample_sub['QuestionId_Answer']].reset_index()
submission.to_csv('submission.csv', index=False)

print("Ensemble submission created successfully!")
print(submission.head())