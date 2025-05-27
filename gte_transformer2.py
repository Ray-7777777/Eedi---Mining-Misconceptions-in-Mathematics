import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import re

# Configuration optimisée
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/kaggle/input/modified-gte-base-weights/gte-base-weights/gte-base_trained_model_version2'
BATCH_SIZE = 8
TOP_K = 27
UNSEEN_BOOST = 3.4
MIN_SIMILARITY = 0.165
CONSTRUCT_BOOST = 1.28

# Chargement du modèle
print("🚀 Chargement du modèle optimisé...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
model.eval()

# Nettoyage du texte mathématique (version simplifiée)
def clean_math_text(text):
    text = str(text)
    text = re.sub(r'([\+\-\*\/=<>\(\)])', r' \1 ', text)  # Espaces autour des opérateurs
    text = re.sub(r'\s+', ' ', text)  # Suppression des espaces multiples
    return text.strip()

# Chargement des données
print("\n📊 Chargement et analyse des données...")
train_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
misconceptions_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')

# Statistiques des méconceptions
construct_miscon = defaultdict(lambda: defaultdict(int))
misconception_freq = defaultdict(int)

for _, row in train_df.iterrows():
    for option in ['A', 'B', 'C', 'D']:
        mid = row[f'Misconception{option}Id']
        if pd.notna(mid):
            construct_miscon[row['ConstructId']][mid] += 1
            misconception_freq[mid] += 1

seen_misconceptions = set(misconception_freq.keys())

# Préparation des données
def prepare_data(df):
    data = []
    for _, row in df.iterrows():
        for option in ['A', 'B', 'C', 'D']:
            if row['CorrectAnswer'] != option:
                context = (
                    "Math Problem: " + clean_math_text(row['QuestionText']) + "\n" +
                    "Wrong Option " + option + ": " + clean_math_text(row[f'Answer{option}Text']) + "\n" +
                    "Subject: " + str(row['SubjectName']) + "\n" +
                    "Construct: " + str(row['ConstructName']) + "\n" +
                    "Predict the exact misconception:"
                )
                data.append({
                    'QuestionId_Answer': f"{row['QuestionId']}_{option}",
                    'ConstructId': row['ConstructId'],
                    'context': context
                })
    return pd.DataFrame(data)

test_data = prepare_data(test_df)

# Génération des embeddings
@torch.no_grad()
def generate_embeddings(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)

# Calcul des embeddings
print("\n🧠 Génération des embeddings...")
test_embeddings = generate_embeddings(test_data['context'].tolist())

misconception_texts = [
    "Math Misconception: " + clean_math_text(row['MisconceptionName']) + "\n" +
    "Description:"
    for _, row in misconceptions_df.iterrows()
]
misconception_embeddings = generate_embeddings(misconception_texts)

# Calcul des similarités avec boosts
def calculate_scores(test_emb, miscon_emb):
    sim = cosine_similarity(test_emb, miscon_emb)
    
    # Boost par fréquence
    freq_weights = np.array([misconception_freq.get(mid, 0) for mid in misconceptions_df['MisconceptionId']])
    freq_weights = np.log1p(freq_weights) + 1.0
    
    # Boost par construct
    construct_weights = np.ones_like(sim)
    for i, construct in enumerate(test_data['ConstructId']):
        if construct in construct_miscon:
            for j, mid in enumerate(misconceptions_df['MisconceptionId']):
                if mid in construct_miscon[construct]:
                    construct_weights[i,j] = CONSTRUCT_BOOST
    
    # Application des boosts
    sim = sim * freq_weights * construct_weights
    
    # Boost pour non-vus
    unseen_mask = np.array([mid not in seen_misconceptions for mid in misconceptions_df['MisconceptionId']])
    sim[:, unseen_mask] *= UNSEEN_BOOST
    
    # Filtrage
    sim[sim < MIN_SIMILARITY] = 0
    return sim

scores = calculate_scores(test_embeddings, misconception_embeddings)

# Génération des prédictions
def generate_predictions(sim_matrix):
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]
    predictions = []
    
    for i, indices in enumerate(top_indices):
        valid_preds = []
        seen = set()
        
        for idx in indices:
            if sim_matrix[i, idx] == 0:
                continue
                
            mid = str(misconceptions_df.iloc[idx]['MisconceptionId'])
            if mid not in seen:
                valid_preds.append(mid)
                seen.add(mid)
        
        predictions.append(' '.join(valid_preds) if valid_preds else '1')
    return predictions

predictions = generate_predictions(scores)

# Création de la soumission
submission = pd.DataFrame({
    'QuestionId_Answer': test_data['QuestionId_Answer'],
    'MisconceptionId': predictions
})

sample_sub = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv')
final_sub = sample_sub[['QuestionId_Answer']].merge(submission, on='QuestionId_Answer', how='left')
final_sub['MisconceptionId'] = final_sub['MisconceptionId'].fillna('1')

final_sub.to_csv('submission.csv', index=False)
print("\n Soumission optimisée prête!")