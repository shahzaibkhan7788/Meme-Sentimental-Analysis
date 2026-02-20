import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RomanUrduTextProcessor:
    """Specialized text processor for Roman Urdu/Hindi-English code-mixed text."""
    
    def __init__(self):
        self.roman_urdu_patterns = {
            'common_words': [
                'kyu', 'kyun', 'kya', 'kaisa', 'kaise', 'hai', 'hain', 'ho', 'raha', 'rahi',
                'gaya', 'gayi', 'jata', 'jati', 'kar', 'ki', 'ko', 'se', 'ne', 'par', 'mein',
                'bhi', 'toh', 'yeh', 'woh', 'mera', 'meri', 'apna', 'apni', 'tumhara', 'humara',
                'acha', 'accha', 'bura', 'khush', 'dukhi', 'masoom', 'bakwas', 'gareeb', 'ameer',
                'dil', 'pyar', 'mohabbat', 'dosti', 'yaar', 'dost', 'saath', 'sath', 'ke'
            ],
            'laughter': ['haha', 'hehe', 'lol', 'lmao', 'rofl', 'hahaha']
        }
        
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.user_mention_pattern = re.compile(r'@\w+')

    def clean_meme_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        cleaned = self.url_pattern.sub('', text)
        cleaned = self.user_mention_pattern.sub('', cleaned)
        cleaned = re.sub(r'(.)\1{3,}', r'\1\1', cleaned)
        
        emojis = self.emoji_pattern.findall(text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if emojis:
            cleaned = f"{cleaned} {' '.join(emojis)}"
        return cleaned

    def detect_language_mix(self, text: str) -> Dict[str, float]:
        if not text: return {'english': 0.0, 'roman_urdu': 0.0, 'other': 1.0}
        words = text.lower().split()
        if not words: return {'english': 0.0, 'roman_urdu': 0.0, 'other': 1.0}
        
        ru_count = sum(1 for w in words if any(p in w for p in self.roman_urdu_patterns['common_words']))
        en_count = sum(1 for w in words if re.match(r'^[a-z]+$', w) and len(w) > 2)
        total = len(words)
        
        return {
            'english': en_count / total,
            'roman_urdu': ru_count / total,
            'other': (total - en_count - ru_count) / total
        }

class XLMRobertaFeatureExtractor:
    """XLM-RoBERTa feature extractor for code-mixed meme text."""

    def __init__(self, model_name='xlm-roberta-base', device=None, max_length=128):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.processor = RomanUrduTextProcessor()
        self.feature_dim = self.model.config.hidden_size

    def extract_features_batch(self, texts: List[str], batch_size=32) -> Tuple[np.ndarray, List[Dict]]:
        all_embeddings = []
        all_lang_mix = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Features"):
            batch = texts[i:i+batch_size]
            cleaned_batch = [self.processor.clean_meme_text(t) for t in batch]
            
            inputs = self.tokenizer(
                cleaned_batch, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token (index 0)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
            
            for t in batch:
                all_lang_mix.append(self.processor.detect_language_mix(t))
                
        return np.vstack(all_embeddings), all_lang_mix

def create_xlmr_features_pipeline(split='train', config_path='config.yml', text_column='Description'):
    """Extract XLM-R features for a dataset split defined in config.yml."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['paths'][f"{split}_csv"]
    output_path = config['paths'][f"xlmr_{split}_features"]

    df = pd.read_csv(csv_path)
    texts = df[text_column].fillna("").astype(str).tolist()

    extractor = XLMRobertaFeatureExtractor(
        model_name=config['text_settings']['model_name'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=config['text_settings']['max_length']
    )

    embeddings, lang_stats = extractor.extract_features_batch(texts, batch_size=config['text_settings']['batch_size'])

    feat_cols = [f'xlmr_feat_{i}' for i in range(embeddings.shape[1])]
    features_df = pd.DataFrame(embeddings, columns=feat_cols)

    lang_df = pd.DataFrame(lang_stats)
    lang_df.columns = [f'lang_{c}' for c in lang_df.columns]

    final_df = pd.concat([df.reset_index(drop=True), features_df, lang_df], axis=1)
    final_df.to_csv(output_path, index=False)

    logger.info(f"✅ Saved {split} XLM-R features to: {output_path} ({final_df.shape})")
    return final_df

if __name__ == "__main__":
    logger.info("--- Extracting XLM-R features (train split) ---")
    create_xlmr_features_pipeline(split='train')

    logger.info("--- Extracting XLM-R features (test split) ---")
    create_xlmr_features_pipeline(split='test')
