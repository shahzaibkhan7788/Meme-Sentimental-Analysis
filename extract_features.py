"""Convenience runners for feature extraction."""

from xlmr_text_processor import create_xlmr_features_pipeline
from Visual_features_extraction import create_vit_features_pipeline


if __name__ == "__main__":
    # Text features
    create_xlmr_features_pipeline(split="train")
    create_xlmr_features_pipeline(split="test")

    # Visual features
    create_vit_features_pipeline(split="train")
    create_vit_features_pipeline(split="test")
