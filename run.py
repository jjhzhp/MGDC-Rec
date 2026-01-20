# coding=utf-8

import argparse
import time
import os
import logging
import yaml
import datetime
import torch
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader

from dataset import POIDataset, collate_fn_4sq
from model import MGDC
from model_components import NegativeSamplingLoss, FGM
from utils import batch_performance, smart_memory_management
from config import get_dataset_config, get_dataset_paths, EVALUATION_K_VALUES


# Clear cache at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Configure cuDNN for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# parse argument
parser = argparse.ArgumentParser()

# Basic Configuration
parser.add_argument('--dataset', default="NYC", help='NYC/TKY')
parser.add_argument('--seed', type=int, default=2023, help='Random seed')
parser.add_argument('--deviceID', type=int, default=0, help='GPU device ID')
parser.add_argument('--save_dir', type=str, default="logs", help='Directory to save logs and models')

# Training Hyperparameters
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=200, help='Input batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--decay', type=float, default=1e-3, help='Weight decay (L2 regularization)')
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')

# Model Architecture
parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--num_mv_layers', type=int, default=3, help='Number of multi-view hypergraph layers')
parser.add_argument('--num_di_layers', type=int, default=3, help='Number of directed hypergraph layers')

# Contrastive Learning
parser.add_argument('--lambda_cl', type=float, default=0.05, help='Weight of contrastive loss')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive learning')

# Loss Functions
parser.add_argument('--focal_alpha', type=float, default=0.5, help='Focal Loss alpha parameter')
parser.add_argument('--focal_gamma', type=float, default=1.5, help='Focal Loss gamma parameter')
parser.add_argument('--num_neg_samples', type=int, default=8, help='Number of negative samples')

# Adversarial Training
parser.add_argument('--adv_epsilon', type=float, default=1.0, help='Adversarial perturbation magnitude')

args = parser.parse_args()

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# set device gpu/cpu
device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")

# set save_dir
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
current_save_dir = os.path.join(args.save_dir, current_time)

# create current save_dir
os.mkdir(current_save_dir)

# Setup logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(current_save_dir, f"log_training.txt"),
                    filemode='w+')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('matplotlib.font_manager').disabled = True

# Save run settings
args_filename = args.dataset + '_args.yaml'
with open(os.path.join(current_save_dir, args_filename), 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)


def main():
    # Parse Arguments
    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(device))
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    dataset_paths = get_dataset_paths(args.dataset)
    
    NUM_USERS = dataset_config['num_users']
    NUM_POIS = dataset_config['num_pois']
    PADDING_IDX = dataset_config['padding_idx']
    
    logging.info(f"Dataset: {dataset_config['dataset_name']}")
    logging.info(f"  Users: {NUM_USERS}, POIs: {NUM_POIS}, Padding Index: {PADDING_IDX}")

    # Load Dataset
    logging.info("2. Load Dataset")
    train_dataset = POIDataset(data_filename=dataset_paths['train_data'],
                               pois_coos_filename=dataset_paths['pois_coos'],
                               num_users=NUM_USERS,
                               num_pois=NUM_POIS,
                               padding_idx=PADDING_IDX,
                               args=args,
                               device=device)

    test_dataset = POIDataset(data_filename=dataset_paths['test_data'],
                              pois_coos_filename=dataset_paths['pois_coos'],
                              num_users=NUM_USERS,
                              num_pois=NUM_POIS,
                              padding_idx=PADDING_IDX,
                              args=args,
                              device=device)

    # 3. Construct DataLoader
    logging.info("3. Construct DataLoader")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX))

    # Load Model
    logging.info("4. Load Model")
    # Add contrastive learning related parameters to args
    args.contrastive_temperature = getattr(args, 'temperature', 0.1)
    args.contrastive_weight = getattr(args, 'lambda_cl', 0.1)
    
    model = MGDC(NUM_USERS, NUM_POIS, args, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # Loss function: Negative Sampling Enhanced Loss (forced)
    criterion = NegativeSamplingLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        num_neg_samples=args.num_neg_samples,
    ).to(device)
    logging.info(f"Using Negative Sampling Enhanced Loss:")
    logging.info(f"  Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    
    # Adversarial Training: FGM (forced)
    fgm = FGM(model, epsilon=args.adv_epsilon, emb_names=['embedding'])
    logging.info(f"Using Adversarial Training (FGM):")
    logging.info(f"  Perturbation magnitude (epsilon): {args.adv_epsilon}")
    logging.info(f"  Targeting: POI and User embeddings")
    
    # Contrastive Learning
    logging.info(f"Contrastive Learning (Mixed Loss):")
    logging.info(f"  Temperature: {args.contrastive_temperature}")
    logging.info(f"  Weight (lambda_cl): {args.lambda_cl}")
    logging.info(f"  Mixed Loss Weights:")
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # Train
    logging.info("5. Start Training")
    Ks_list = EVALUATION_K_VALUES
    final_results = {"Rec1": 0.0, "Rec5": 0.0, "Rec10": 0.0, "Rec20": 0.0,
                     "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0, "NDCG20": 0.0,
                     }

    monitor_loss = float('inf')
    best_test_rec5 = 0.0
    for epoch in range(args.num_epochs):
        logging.info("================= Epoch {}/{} =================".format(epoch, args.num_epochs))
        start_time = time.time()
        model.train()

        train_loss = 0.0

        # to save recall and ndcg results
        train_recall_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        for idx, batch in enumerate(train_dataloader):
            logging.info("Train. Batch {}/{}".format(idx, len(train_dataloader)))
            
            optimizer.zero_grad()

            # 1. Normal forward pass
            predictions, contrastive_loss = model(train_dataset, batch)
            
            # Compute NegativeSamplingLoss (returns 3 values: total_loss, focal_loss, ranking_loss)
            main_loss, focal_loss, ranking_loss = criterion(predictions, batch["label"].to(device))
            
            # Combine main loss and contrastive loss
            total_loss = main_loss + args.lambda_cl * contrastive_loss
            
            # 2. Backward pass to compute gradients
            total_loss.backward()
            
            # 3. Adversarial Training: Add perturbation
            fgm.attack()
            
            # 4. Forward pass with adversarial examples
            predictions_adv, contrastive_loss_adv = model(train_dataset, batch)
            
            # Compute adversarial loss
            main_loss_adv, focal_loss_adv, ranking_loss_adv = criterion(predictions_adv, batch["label"].to(device))
            total_loss_adv = main_loss_adv + args.lambda_cl * contrastive_loss_adv
            
            # 5. Backward pass for adversarial loss
            total_loss_adv.backward()
            
            # 6. Restore embeddings
            fgm.restore()
            
            # 7. Update parameters
            optimizer.step()
            
            # Log both clean and adversarial losses
            logging.info("Train. Clean: {:.4f} (F: {:.4f}, R: {:.4f}, CL: {:.4f}) | Adv: {:.4f} (F: {:.4f}, R: {:.4f}, CL: {:.4f})".format(
                main_loss.item(), focal_loss.item(), ranking_loss.item(), contrastive_loss.item(),
                main_loss_adv.item(), focal_loss_adv.item(), ranking_loss_adv.item(), contrastive_loss_adv.item()))
            
            train_loss += (total_loss.item() + total_loss_adv.item()) / 2.0

            for k in Ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                col_idx = Ks_list.index(k)
                train_recall_array[idx, col_idx] = recall
                train_ndcg_array[idx, col_idx] = ndcg
            
            del predictions

        # Smart memory cleanup after training epoch
        if smart_memory_management():
            logging.info("GPU memory cache cleared (usage > 85%)")

        logging.info("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
        logging.info("Training loss: {:.4f}".format(train_loss / len(train_dataloader)))
        logging.info("Training Epoch {}/{} results:".format(epoch, args.num_epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            logging.info("Recall@{}: {:.4f}".format(k, np.mean(train_recall_array[:, col_idx])))
            logging.info("NDCG@{}: {:.4f}".format(k, np.mean(train_ndcg_array[:, col_idx])))
        logging.info("\n")

        logging.info("Testing")
        test_loss = 0.0
        test_recall_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
        test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):

                logging.info("Test. Batch {}/{}".format(idx, len(test_dataloader)))

                predictions, contrastive_loss = model(test_dataset, batch)
                
                # calculate main task loss (NegativeSamplingLoss returns 3 values)
                main_loss, focal_loss, ranking_loss = criterion(predictions, batch["label"].to(device))
                
                loss = main_loss + args.lambda_cl * contrastive_loss
                logging.info("Test. main_loss: {:.4f} (F: {:.4f}, R: {:.4f}); contrastive_loss: {:.4f}; total_loss: {:.4f}".format(
                    main_loss.item(), focal_loss.item(), ranking_loss.item(), contrastive_loss.item(), loss.item()))

                test_loss += loss.item()

                for k in Ks_list:
                    recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                    col_idx = Ks_list.index(k)
                    test_recall_array[idx, col_idx] = recall
                    test_ndcg_array[idx, col_idx] = ndcg

        logging.info("Testing finishes")
        logging.info("Testing loss: {}".format(test_loss / len(test_dataloader)))
        logging.info("Testing results:")
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            recall = np.mean(test_recall_array[:, col_idx])
            ndcg = np.mean(test_ndcg_array[:, col_idx])
            logging.info("Recall@{}: {:.4f}".format(k, recall))
            logging.info("NDCG@{}: {:.4f}".format(k, ndcg))

        # Check monitor loss and monitor score for updating
        monitor_loss = min(monitor_loss, test_loss)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # update best_test_rec5
        test_recall5 = np.mean(test_recall_array[:, 1])
        if test_recall5 > best_test_rec5:
            best_test_rec5 = test_recall5
            logging.info("Update test results and save model at epoch{}".format(epoch))

            # define saved_model_path
            saved_model_path = os.path.join(current_save_dir, "{}.pt".format(args.dataset))
            torch.save(model.state_dict(), saved_model_path)

        # update best result
        for k in Ks_list:
            if k == 1:
                final_results["Rec1"] = max(final_results["Rec1"], np.mean(test_recall_array[:, 0]))
                final_results["NDCG1"] = max(final_results["NDCG1"], np.mean(test_ndcg_array[:, 0]))

            elif k == 5:
                final_results["Rec5"] = max(final_results["Rec5"], np.mean(test_recall_array[:, 1]))
                final_results["NDCG5"] = max(final_results["NDCG5"], np.mean(test_ndcg_array[:, 1]))

            elif k == 10:
                final_results["Rec10"] = max(final_results["Rec10"], np.mean(test_recall_array[:, 2]))
                final_results["NDCG10"] = max(final_results["NDCG10"], np.mean(test_ndcg_array[:, 2]))

            elif k == 20:
                final_results["Rec20"] = max(final_results["Rec20"], np.mean(test_recall_array[:, 3]))
                final_results["NDCG20"] = max(final_results["NDCG20"], np.mean(test_ndcg_array[:, 3]))
        logging.info("==================================\n\n")

    logging.info("6. Final Results")
    formatted_dict = {key: f"{value:.4f}" for key, value in final_results.items()}
    logging.info(formatted_dict)
    logging.info("\n")


if __name__ == '__main__':
    main()

