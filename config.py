# coding=utf-8
"""
Configuration file for POI recommendation system
"""

# Dataset configurations
DATASET_CONFIG = {
    "TKY": {
        "num_users": 2173,
        "num_pois": 7038,
        "dataset_name": "Tokyo"
    },
    "NYC": {
        "num_users": 834,
        "num_pois": 3835,
        "dataset_name": "New York City"
    }
}

# Model default hyperparameters
DEFAULT_MODEL_CONFIG = {
    "emb_dim": 128,
    "dropout": 0.4,
    "num_mv_layers": 3,
    "num_di_layers": 3,
    "num_seq_heads": 4,
    "interval": 100,  # Distance interval bins
}

# Training default hyperparameters
DEFAULT_TRAINING_CONFIG = {
    "num_epochs": 30,
    "batch_size": 200,
    "lr": 1e-3,
    "decay": 1e-3,
    "lr_scheduler_factor": 0.1,
}

# Contrastive learning defaults
DEFAULT_CL_CONFIG = {
    "lambda_cl": 0.05,
    "temperature": 0.1,
    "contrastive_temperature": 0.1,
}

# Loss function defaults
DEFAULT_LOSS_CONFIG = {
    "focal_alpha": 0.5,
    "focal_gamma": 1.5,
    "num_neg_samples": 8,
    "neg_weight": 0.2,
    "hard_ratio": 0.25,
    "popular_ratio": 0.25,
}

# Adversarial training defaults
DEFAULT_ADV_CONFIG = {
    "adv_epsilon": 1.0,
}

# Evaluation metrics
EVALUATION_K_VALUES = [1, 5, 10, 20]

# Temporal embedding config
TEMPORAL_CONFIG = {
    "max_position_embeddings": 1500,
}


def get_dataset_config(dataset_name):
    """
    Get configuration for a specific dataset
    
    Args:
        dataset_name: Dataset identifier (e.g., 'NYC', 'TKY')
    
    Returns:
        dict: Dataset configuration including num_users, num_pois, padding_idx
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(DATASET_CONFIG.keys())}"
        )
    
    config = DATASET_CONFIG[dataset_name].copy()
    config['padding_idx'] = config['num_pois']  # Padding index is num_pois
    
    return config


def get_dataset_paths(dataset_name):
    """
    Get file paths for a specific dataset
    
    Args:
        dataset_name: Dataset identifier (e.g., 'NYC', 'TKY')
    
    Returns:
        dict: Paths to train/test data and POI coordinates
    """
    base_dir = f"datasets/{dataset_name}"
    
    return {
        'train_data': f"{base_dir}/train.txt",
        'test_data': f"{base_dir}/test.txt",
        'pois_coos': f"{base_dir}/{dataset_name}_poi_coords.pkl",
    }

