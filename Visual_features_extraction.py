import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemeViTFeatureExtractor:
    """
    Vision Transformer feature extractor for meme images.
    Extracts deep visual features using pretrained ViT models.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224', 
                 device=None, image_size=224):
        """
        Initialize ViT feature extractor
        
        Args:
            model_name: Pretrained ViT model name
            device: 'cuda' or 'cpu' (auto-detected if None)
            image_size: Input image size (default 224x224 for ViT-Base)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.image_size = image_size
        
        logger.info(f"Initializing ViT Feature Extractor on {self.device}")
        logger.info(f"Model: {model_name}")
        
        # Initialize feature extractor and model
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.feature_extractor.image_mean,
                std=self.feature_extractor.image_std
            )
        ])
        
        # Store feature dimensions
        self.feature_dim = self.model.config.hidden_size  # 768 for ViT-Base
        logger.info(f"Feature dimension: {self.feature_dim}")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            return None
    
    def extract_features_single(self, image_path, return_tensor=True):
        """
        Extract features from single image
        
        Args:
            image_path: Path to image file
            return_tensor: Return PyTorch tensor instead of numpy
            
        Returns:
            Feature vector (768-dim for ViT-Base)
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                return None
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Extract features (no gradient tracking)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # Get [CLS] token embedding (last hidden state)
                # Shape: (batch_size, sequence_length, hidden_size)
                last_hidden_state = outputs.last_hidden_state
                
                # Use [CLS] token for image representation
                # CLS token is at position 0
                features = last_hidden_state[:, 0, :]  # Shape: (1, 768)
                
                # Also get pooled output if available
                pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else features
            
            # Convert to numpy if requested
            if not return_tensor:
                features = features.cpu().numpy().squeeze()
                pooled_output = pooled_output.cpu().numpy().squeeze()
            
            return {
                'features': features,
                'pooled_output': pooled_output,
                'all_tokens': last_hidden_state if return_tensor else last_hidden_state.cpu().numpy()
            }
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_features_batch(self, image_paths, batch_size=32, save_dir=None):
        """
        Extract features from multiple images in batches
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            save_dir: Directory to save features (optional)
            
        Returns:
            Dictionary with features for all images
        """
        all_features = {}
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving features to {save_dir}")
        
        # Process in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        with tqdm(total=len(image_paths), desc="Extracting ViT features") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]
                
                batch_tensors = []
                valid_paths = []
                
                # Load and preprocess batch
                for img_path in batch_paths:
                    tensor = self.preprocess_image(img_path)
                    if tensor is not None:
                        batch_tensors.append(tensor)
                        valid_paths.append(img_path)
                
                if not batch_tensors:
                    continue
                
                # Stack batch
                batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    batch_features = outputs.last_hidden_state[:, 0, :]  # CLS tokens
                
                # Store features
                for i, img_path in enumerate(valid_paths):
                    img_id = Path(img_path).stem
                    features_np = batch_features[i].cpu().numpy()
                    
                    all_features[img_id] = features_np
                    
                    # Save to disk if requested
                    if save_dir:
                        save_path = os.path.join(save_dir, f"{img_id}_vit.npy")
                        np.save(save_path, features_np)
                
                pbar.update(len(batch_paths))
        
        logger.info(f"Successfully extracted features from {len(all_features)} images")
        return all_features
    
    def extract_patch_features(self, image_path, layer_index=-1):
        """
        Extract patch-level features (for visualization or attention analysis)
        
        Args:
            image_path: Path to image
            layer_index: Which transformer layer to use (-1 for last layer)
            
        Returns:
            Patch features and attention weights
        """
        try:
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                return None
            
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                # Get all hidden states and attentions
                outputs = self.model(image_tensor, output_attentions=True, output_hidden_states=True)
                
                # Get patch embeddings (excluding CLS token)
                # For ViT-Base: 197 tokens (1 CLS + 196 patches)
                last_hidden_state = outputs.hidden_states[layer_index]
                patch_features = last_hidden_state[:, 1:, :]  # Remove CLS token
                
                # Get attention weights from last layer
                attentions = outputs.attentions[-1]  # Last layer attention
                
                # Average attention across heads
                avg_attention = attentions.mean(dim=1)
            
            return {
                'patch_features': patch_features.squeeze(),
                'attention_weights': avg_attention.squeeze(),
                'cls_token': last_hidden_state[:, 0, :].squeeze()
            }
            
        except Exception as e:
            logger.error(f"Error extracting patch features: {e}")
            return None
    
    def extract_features_from_dataframe(self, df, image_dir, image_column='Meme Number',
                                        save_path=None, batch_size=32):
        """
        Extract features for images referenced in DataFrame
        
        Args:
            df: DataFrame with image references
            image_dir: Directory containing images
            image_column: Column with image identifiers
            save_path: Path to save features DataFrame
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with extracted features
        """
        # Build image paths
        image_paths = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            img_id = f"meme_{row[image_column]}"

            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                img_path = os.path.join(image_dir, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    valid_indices.append(idx)
                    break

        
        if not image_paths:
            logger.error(f"No images found in {image_dir}")
            return None
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Extract features
        features_dict = self.extract_features_batch(image_paths, batch_size=batch_size)
        
        # Create features DataFrame
        feature_columns = [f'vit_feat_{i}' for i in range(self.feature_dim)]
        features_data = []
        
        features_data = []
        
        for idx in valid_indices:
            img_id = f"meme_{df.iloc[idx][image_column]}"

            if img_id in features_dict:
                features_data.append(features_dict[img_id])
            else:
                features_data.append(np.zeros(self.feature_dim))

        
        features_df = pd.DataFrame(features_data, columns=feature_columns)
        features_df.index = valid_indices
        
        # Merge with original DataFrame
        result_df = df.copy()
        for col in feature_columns:
            result_df[col] = np.nan
        
        result_df.loc[valid_indices, feature_columns] = features_df.values
        
        # Save if requested
        if save_path:
            result_df.to_csv(save_path, index=False)
            logger.info(f"Saved features to {save_path}")
        
        # Log statistics
        extracted_count = len([f for f in features_data if not np.all(f == 0)])
        logger.info(f"Extracted features for {extracted_count}/{len(df)} memes")
        
        return result_df
    
    def save_features(self, features_dict, output_path):
        """Save extracted features to disk"""
        np.save(output_path, features_dict)
        logger.info(f"Features saved to {output_path}")
    
    def load_features(self, input_path):
        """Load previously saved features"""
        features_dict = np.load(input_path, allow_pickle=True).item()
        logger.info(f"Loaded {len(features_dict)} features from {input_path}")
        return features_dict


class EnhancedViTFeatureExtractor(MemeViTFeatureExtractor):
    """
    Enhanced ViT extractor with multiple feature extraction strategies
    and model variants support.
    """
    
    def __init__(self, model_variant='base', use_dino=False, **kwargs):
        """
        Initialize enhanced extractor with different ViT variants
        
        Args:
            model_variant: 'base', 'large', 'huge', or 'small'
            use_dino: Use DINO-pretrained ViT (better for unsupervised features)
        """
        # Map variants to model names
        model_mapping = {
            'base': 'google/vit-base-patch16-224',
            'large': 'google/vit-large-patch16-224',
            'huge': 'google/vit-huge-patch14-224-in21k',
            'small': 'google/vit-small-patch16-224',
        }
        
        if use_dino:
            model_name = 'facebook/dino-vitb16'
            logger.info("Using DINO-pretrained ViT for self-supervised features")
        else:
            model_name = model_mapping.get(model_variant, model_mapping['base'])
        
        super().__init__(model_name=model_name, **kwargs)
        
        # Additional initialization for enhanced features
        self.model_variant = model_variant
        self.use_dino = use_dino
    
    def extract_multilevel_features(self, image_path, layers=[-1, -2, -3]):
        """
        Extract features from multiple transformer layers
        
        Args:
            image_path: Path to image
            layers: List of layer indices to extract features from
            
        Returns:
            Dictionary with features from specified layers
        """
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor, output_hidden_states=True)
            
            features = {}
            for layer_idx in layers:
                # Get hidden state from specified layer
                hidden_state = outputs.hidden_states[layer_idx]
                
                # Extract CLS token
                cls_features = hidden_state[:, 0, :]
                
                # Store features
                features[f'layer_{layer_idx}'] = cls_features.cpu().numpy().squeeze()
            
            # Also extract patch features from last layer for spatial understanding
            last_hidden = outputs.hidden_states[-1]
            patch_features = last_hidden[:, 1:, :]  # All patches except CLS
            
            features['patch_features'] = patch_features.cpu().numpy().squeeze()
        
        return features
    
    def compute_spatial_attention_map(self, image_path, head_index=0):
        """
        Compute spatial attention map for visualization
        
        Args:
            image_path: Path to image
            head_index: Which attention head to visualize
            
        Returns:
            Attention map resized to original image dimensions
        """
        from PIL import Image
        import torch.nn.functional as F
        
        # Load original image for dimensions
        orig_image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = orig_image.size
        
        # Extract attention weights
        result = self.extract_patch_features(image_path)
        if result is None:
            return None
        
        attention_weights = result['attention_weights']
        
        # Get attention for specific head
        if len(attention_weights.shape) == 3:
            # Shape: (num_heads, num_tokens, num_tokens)
            head_attention = attention_weights[head_index]
            
            # CLS token attention to patches (first row, excluding self)
            cls_to_patches = head_attention[0, 1:]  # Shape: (196,)
        else:
            cls_to_patches = attention_weights[0, 1:]
        
        # Reshape to patch grid (14x14 for ViT-Base/16)
        grid_size = int(np.sqrt(len(cls_to_patches)))
        attention_map = cls_to_patches.reshape(grid_size, grid_size)
        
        # Resize to original image dimensions
        attention_map_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0)
        attention_map_resized = F.interpolate(
            attention_map_tensor,
            size=(orig_height, orig_width),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # Normalize
        attention_map_resized = (attention_map_resized - attention_map_resized.min()) / \
                               (attention_map_resized.max() - attention_map_resized.min())
        
        return attention_map_resized


# Utility functions for the project
def create_vit_features_pipeline(split='train', config_path='config.yml'):
    """Extract ViT features for a dataset split defined in config.yml."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['paths'][f"{split}_csv"]
    image_dir = config['paths'][f"{split}_image_dir"]
    output_path = config['paths'][f"vit_{split}_features"]

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} memes from {csv_path}")

    extractor = EnhancedViTFeatureExtractor(
        model_variant=config['vit_settings']['model_variant'],
        use_dino=config['vit_settings']['use_dino'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    features_df = extractor.extract_features_from_dataframe(
        df=df,
        image_dir=image_dir,
        image_column='Meme Number',
        save_path=output_path,
        batch_size=config['vit_settings']['batch_size']
    )

    return features_df


def visualize_attention(image_path, extractor=None):
    """
    Visualize ViT attention on meme image
    
    Args:
        image_path: Path to meme image
        extractor: ViTFeatureExtractor instance
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    if extractor is None:
        extractor = MemeViTFeatureExtractor()
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get attention map
    attention_map = extractor.compute_spatial_attention_map(image_path)
    
    if attention_map is None:
        logger.error("Could not compute attention map")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Meme')
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('ViT Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attention_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    logger.info("--- Extracting ViT features (train split) ---")
    train_df = create_vit_features_pipeline(split='train')
    if train_df is not None:
        logger.info(f"Saved train features: {train_df.shape}")

    logger.info("--- Extracting ViT features (test split) ---")
    test_df = create_vit_features_pipeline(split='test')
    if test_df is not None:
        logger.info(f"Saved test features: {test_df.shape}")

    
