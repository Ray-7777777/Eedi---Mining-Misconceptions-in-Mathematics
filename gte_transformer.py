import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

# Configuration 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/kaggle/input/modified-gte-base-weights/gte-base-weights/gte-base_trained_model_version2'
BATCH_SIZE = 8  
TOP_K = 25
UNSEEN_BOOST = 3.0  

# Chargement du modèle 
print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    local_files_only=True, 
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    MODEL_PATH, 
    local_files_only=True, 
    trust_remote_code=True
).to(DEVICE).eval()

# Chargement des données 
train_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
misconceptions_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
sample_sub = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv')

# Identification des méconceptions vues 
seen_misconceptions = set()
for col in ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']:
    seen_misconceptions.update(train_df[col].dropna().unique())

# Préparation des données de test
test_data = []
for _, row in test_df.iterrows():
    for option in ['A', 'B', 'C', 'D']:
        test_data.append({
            'QuestionId_Answer': f"{row['QuestionId']}_{option}",
            'QuestionText': row['QuestionText'],
            'CorrectAnswer': row['CorrectAnswer'],
            'Option': option,
            'ConstructName': row['ConstructName'],
            'SubjectName': row['SubjectName'],
            'OptionText': row[f'Answer{option}Text']  # Ajout pour la préparation du texte
        })
test_df_processed = pd.DataFrame(test_data)

# Préparation du texte 
def prepare_text(row):
    return (
        f"Question: {row['QuestionText']}\n"
        f"Option {row['Option']}: {row['OptionText']}\n"
        f"Subject: {row['SubjectName']}\n"
        f"Construct: {row['ConstructName']}"
    )

# Génération des embeddings avec pooling du premier token 
def get_embeddings(texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=512,  # Longueur originale
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Utilisation du premier token 
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

print("\nGénération des embeddings...")
test_texts = test_df_processed.apply(prepare_text, axis=1).tolist()
test_embeddings = get_embeddings(test_texts)

# Préparation des textes de méconceptions 
misconception_texts = misconceptions_df['MisconceptionName'].apply(
    lambda x: f"Misconception: {x}"
).tolist()
misconception_embeddings = get_embeddings(misconception_texts)

# Calcul des similarités 
similarities = cosine_similarity(test_embeddings, misconception_embeddings)

# Application du boost aux méconceptions non vues
unseen_indices = [i for i, mid in enumerate(misconceptions_df['MisconceptionId']) 
                 if mid not in seen_misconceptions]
similarities[:, unseen_indices] *= UNSEEN_BOOST

# Sélection des prédictions top K avec vérification des doublons
top_indices = np.argsort(-similarities, axis=1)[:, :TOP_K]

submission_data = []
for i, row in test_df_processed.iterrows():
    seen_ids = set()
    top_misconceptions = []
    for idx in top_indices[i]:
        misconception_id = str(misconceptions_df.iloc[idx]['MisconceptionId'])
        if misconception_id not in seen_ids:
            top_misconceptions.append(misconception_id)
            seen_ids.add(misconception_id)
        if len(top_misconceptions) >= TOP_K:
            break
    
    submission_data.append({
        'QuestionId_Answer': row['QuestionId_Answer'],
        'MisconceptionId': ' '.join(top_misconceptions)
    })

# Création de la soumission
submission_df = pd.DataFrame(submission_data)

# Garantir le bon format comme dans l'échantillon
final_submission = sample_sub[['QuestionId_Answer']].merge(
    submission_df, 
    on='QuestionId_Answer', 
    how='left'
).fillna('1')  # Remplissage par défaut si nécessaire

# Sauvegarde
submission_path = 'submission.csv'
final_submission.to_csv(submission_path, index=False)
print(f"Soumission sauvegardée dans {submission_path}")