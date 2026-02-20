import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import os
import yaml

# --------------------------------------------------
# BASIC SETUP
# --------------------------------------------------
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 1. DATASET
# --------------------------------------------------
class MultimodalMemeDataset(Dataset):
    def __init__(self, visual_features, text_features, sentiment_labels, intent_labels, topic_labels):
        self.visual_features = torch.FloatTensor(visual_features)
        self.text_features = torch.FloatTensor(text_features)
        self.sentiment_labels = torch.LongTensor(sentiment_labels)
        self.intent_labels = torch.LongTensor(intent_labels)
        self.topic_labels = torch.LongTensor(topic_labels)
        
    def __len__(self):
        return len(self.sentiment_labels)
    
    def __getitem__(self, idx):
        return {
            'visual': self.visual_features[idx],
            'text': self.text_features[idx],
            'sentiment': self.sentiment_labels[idx],
            'intent': self.intent_labels[idx],
            'topic': self.topic_labels[idx]
        }

# --------------------------------------------------
# 2. MODELS
# --------------------------------------------------
class MultimodalFeatureFusion(nn.Module):
    def __init__(self, visual_dim=768, text_dim=768, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, visual_feat, text_feat):
        v_proj = self.visual_proj(visual_feat)
        t_proj = self.text_proj(text_feat)
        combined = torch.cat([v_proj, t_proj], dim=1)
        fused = self.fusion_layer(combined)
        return fused

class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim=512, num_sentiment=3, num_intent=3, num_topic=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256)
        )
        self.sentiment_head = nn.Linear(256, num_sentiment)
        self.intent_head = nn.Linear(256, num_intent)
        self.topic_head = nn.Linear(256, num_topic)
    
    def forward(self, x):
        shared = self.shared(x)
        return {
            'sentiment': self.sentiment_head(shared),
            'intent': self.intent_head(shared),
            'topic': self.topic_head(shared)
        }

# --------------------------------------------------
# 3. DATA LOADING
# --------------------------------------------------
def load_split_data(split, config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vit_path = config["paths"][f"vit_{split}_features"]
    xlmr_path = config["paths"][f"xlmr_{split}_features"]

    vit_df = pd.read_csv(vit_path)
    xlmr_df = pd.read_csv(xlmr_path)

    v_cols = [c for c in vit_df.columns if c.startswith("vit_feat_")]
    t_cols = [c for c in xlmr_df.columns if c.startswith("xlmr_feat_")]

    min_len = min(len(vit_df), len(xlmr_df))
    v_feats = vit_df[v_cols].iloc[:min_len].values
    t_feats = xlmr_df[t_cols].iloc[:min_len].values

    logger.info(f"ViT features shape: {v_feats.shape}")
    logger.info(f"XLM-R features shape: {t_feats.shape}")

    v_feats = np.nan_to_num(v_feats, nan=0.0, posinf=1.0, neginf=-1.0)
    t_feats = np.nan_to_num(t_feats, nan=0.0, posinf=1.0, neginf=-1.0)

    v_feats = (v_feats - v_feats.mean(0)) / (v_feats.std(0) + 1e-8)
    t_feats = (t_feats - t_feats.mean(0)) / (t_feats.std(0) + 1e-8)

    sentiment_map = {"Positive": 0, "Neutral": 1, "Negative": 2}
    sentiment_labels = vit_df["Sentiment"].astype(str).str.strip().map(sentiment_map).fillna(0).astype(int).values

    intent_map = {"Relatable": 0, "Satirical": 1, "Informative": 2}
    intent_labels = vit_df["Intent"].astype(str).str.strip().map(intent_map).fillna(0).astype(int).values

    topic_map = {
        "Entertainment": 0,
        "Culture": 1,
        "Social issues": 2,
        "Social Issues": 2,
        "Politics": 3,
        "Education": 4,
        "Technology": 5
    }
    topic_labels = vit_df["Topic of the Meme"].astype(str).str.strip().map(topic_map).fillna(0).astype(int).values

    return v_feats, t_feats, sentiment_labels, intent_labels, topic_labels


# --------------------------------------------------
# 4. TRAIN + VALIDATION
# --------------------------------------------------
def train_with_validation(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)

    v_feats, t_feats, sentiment_labels, intent_labels, topic_labels = load_split_data("train", config_path)

    logger.info(f"Sentiment class distribution: {np.bincount(sentiment_labels)}")
    logger.info(f"Intent class distribution: {np.bincount(intent_labels)}")
    logger.info(f"Topic class distribution: {np.bincount(topic_labels)}")

    tr_idx, val_idx = train_test_split(
        np.arange(len(sentiment_labels)),
        test_size=config["fusion_training"]["val_split"],
        stratify=sentiment_labels,
        random_state=42
    )

    train_ds = MultimodalMemeDataset(v_feats[tr_idx], t_feats[tr_idx], sentiment_labels[tr_idx], intent_labels[tr_idx], topic_labels[tr_idx])
    val_ds = MultimodalMemeDataset(v_feats[val_idx], t_feats[val_idx], sentiment_labels[val_idx], intent_labels[val_idx], topic_labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=config["fusion_training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["fusion_training"]["batch_size"])

    fusion = MultimodalFeatureFusion(
        visual_dim=v_feats.shape[1],
        text_dim=t_feats.shape[1],
        hidden_dim=config["fusion_training"]["hidden_dim"],
        dropout=config["fusion_training"]["dropout"]
    ).to(DEVICE)
    classifier = MultimodalClassifier(num_sentiment=3, num_intent=3, num_topic=6).to(DEVICE)

    optimizer = torch.optim.Adam(list(fusion.parameters()) + list(classifier.parameters()), lr=config["fusion_training"]["learning_rate"])

    criterion_sentiment = nn.CrossEntropyLoss()
    criterion_intent = nn.CrossEntropyLoss()
    criterion_topic = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(config["fusion_training"]["epochs"]):
        fusion.train()
        classifier.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            visual = batch["visual"].to(DEVICE)
            text = batch["text"].to(DEVICE)
            sentiment = batch["sentiment"].to(DEVICE)
            intent = batch["intent"].to(DEVICE)
            topic = batch["topic"].to(DEVICE)

            fused = fusion(visual, text)
            preds = classifier(fused)

            loss = criterion_sentiment(preds["sentiment"], sentiment) + \
                   criterion_intent(preds["intent"], intent) + \
                   criterion_topic(preds["topic"], topic)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(classifier.parameters()), 1.0)
            optimizer.step()
            train_loss += loss.item()

        fusion.eval()
        classifier.eval()
        val_loss = 0.0
        val_sentiment_preds, val_sentiment_true = [], []
        val_intent_preds, val_intent_true = [], []
        val_topic_preds, val_topic_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                visual = batch["visual"].to(DEVICE)
                text = batch["text"].to(DEVICE)
                sentiment = batch["sentiment"].to(DEVICE)
                intent = batch["intent"].to(DEVICE)
                topic = batch["topic"].to(DEVICE)

                fused = fusion(visual, text)
                preds = classifier(fused)

                loss = criterion_sentiment(preds["sentiment"], sentiment) + \
                       criterion_intent(preds["intent"], intent) + \
                       criterion_topic(preds["topic"], topic)

                val_loss += loss.item()

                val_sentiment_preds.extend(preds["sentiment"].argmax(1).cpu().numpy())
                val_sentiment_true.extend(sentiment.cpu().numpy())
                val_intent_preds.extend(preds["intent"].argmax(1).cpu().numpy())
                val_intent_true.extend(intent.cpu().numpy())
                val_topic_preds.extend(preds["topic"].argmax(1).cpu().numpy())
                val_topic_true.extend(topic.cpu().numpy())

        sentiment_acc = accuracy_score(val_sentiment_true, val_sentiment_preds)
        intent_acc = accuracy_score(val_intent_true, val_intent_preds)
        topic_acc = accuracy_score(val_topic_true, val_topic_preds)

        logger.info(
            f"Epoch {epoch+1}/{config['fusion_training']['epochs']} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Sentiment Acc: {sentiment_acc:.4f} | "
            f"Intent Acc: {intent_acc:.4f} | "
            f"Topic Acc: {topic_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(fusion.state_dict(), os.path.join(config["paths"]["checkpoints_dir"], "best_fusion.pt"))
            torch.save(classifier.state_dict(), os.path.join(config["paths"]["checkpoints_dir"], "best_classifier.pt"))
            logger.info("✅ Best model updated and saved")

    fusion.load_state_dict(torch.load(os.path.join(config["paths"]["checkpoints_dir"], "best_fusion.pt")))
    classifier.load_state_dict(torch.load(os.path.join(config["paths"]["checkpoints_dir"], "best_classifier.pt")))
    logger.info("🎯 Best model loaded (ready for testing)")

    return fusion, classifier

# --------------------------------------------------
# 5. EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    fusion_model, classifier_model = train_with_validation()
