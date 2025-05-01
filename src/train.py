"""Train a Faster R-CNN model for house sparrow detection with MLflow tracking."""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

# Import from custom modules
from data_preparation import prepare_data
from model import faster_rcnn_mob_model_for_n_classes
#from utils import collate_fn  # Assuming this is defined in your utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HouseSparrowDataset(torch.utils.data.Dataset):
    """Dataset for House Sparrow detection."""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file: Path to the CSV file with annotations.
            img_dir: Directory with all the images.
            transform: Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Load bounding box data from bboxes/bounding_boxes.csv
        bbox_file = Path(img_dir).parent / "bboxes" / "bounding_boxes.csv" 
        self.bbox_df = pd.read_csv(bbox_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        """Get image and target for training."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image info
        img_name = self.data_frame.iloc[idx]['Name']
        img_path = os.path.join(self.img_dir, f"{img_name}")
        
        # Read image
        image = self._read_image(img_path)
        
        # Get bounding boxes for this image
        bbox_records = self.bbox_df[self.bbox_df['image_name'] == img_name]
        
        # Prepare target dictionary
        boxes = []
        labels = []
        
        for _, box_row in bbox_records.iterrows():
            # Convert bbox coordinates
            x_min = box_row['bbox_x']
            y_min = box_row['bbox_y']
            width = box_row['bbox_width']
            height = box_row['bbox_height']
            
            # Store as [x_min, y_min, x_max, y_max]
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            
            # House sparrow is class 1 (0 is background)
            labels.append(1)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target
    
    def _read_image(self, path):
        """Read image from file path."""
        from PIL import Image
        import torchvision.transforms.functional as F
        
        # Open image
        img = Image.open(path).convert("RGB")
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        return img_tensor

def collate_fn(batch):
    """
    Custom collate function for object detection data.
    This handles batches with variable number of objects.
    """
    return tuple(zip(*batch))
def train_one_epoch(model, optimizer, data_loader, device):
    """Train the model for one epoch."""
    model.train()
    
    epoch_loss = 0
    epoch_loss_classifier = 0
    epoch_loss_box_reg = 0
    epoch_loss_objectness = 0
    epoch_loss_rpn_box_reg = 0
    
    start_time = time.time()
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track losses
        epoch_loss += losses.item()
        epoch_loss_classifier += loss_dict['loss_classifier'].item()
        epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
        epoch_loss_objectness += loss_dict['loss_objectness'].item()
        epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
    
    time_elapsed = time.time() - start_time
    num_batches = len(data_loader)
    
    train_metrics = {
        'loss': epoch_loss / num_batches,
        'loss_classifier': epoch_loss_classifier / num_batches,
        'loss_box_reg': epoch_loss_box_reg / num_batches,
        'loss_objectness': epoch_loss_objectness / num_batches,
        'loss_rpn_box_reg': epoch_loss_rpn_box_reg / num_batches,
        'time_sec': time_elapsed
    }
    
    return train_metrics


def evaluate(model, data_loader, device, iou_threshold=0.5):
    """Evaluate the model on the validation set."""
    model.eval()
    
    # Tracking metrics
    total_detections = 0
    total_ground_truth = 0
    total_true_positives = 0
    
    val_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Track validation loss if the model supports it
            if model.training:
                model.eval()
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
                model.train()
            
            # Get model predictions
            predictions = model(images)
            
            # Process each image in the batch
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                
                # Filter predictions to include only house sparrows (class 1)
                house_sparrow_mask = pred_labels == 1
                pred_boxes = pred_boxes[house_sparrow_mask]
                pred_scores = pred_scores[house_sparrow_mask]
                
                # Target info
                target_boxes = target['boxes']
                target_labels = target['labels']
                
                # Filter ground truth to include only house sparrows
                gt_mask = target_labels == 1
                target_boxes = target_boxes[gt_mask]
                
                # Count ground truth and predictions
                num_gt = len(target_boxes)
                num_pred = len(pred_boxes)
                
                total_ground_truth += num_gt
                total_detections += num_pred
                
                # If either is empty, continue to next image
                if num_gt == 0 or num_pred == 0:
                    continue
                
                # Calculate IoU between predictions and ground truth
                iou_matrix = box_iou(pred_boxes, target_boxes)
                
                # Find matches above threshold
                matches = iou_matrix >= iou_threshold
                
                # A true positive is a detection that matches a ground truth box
                # Count unique matches to avoid double counting
                for i in range(num_pred):
                    if matches[i].any():
                        total_true_positives += 1
                        # Remove the matched ground truth to avoid double counting
                        matched_gt = matches[i].nonzero(as_tuple=True)[0][0]
                        matches[:, matched_gt] = False
    
    # Calculate metrics
    precision = total_true_positives / total_detections if total_detections > 0 else 0
    recall = total_true_positives / total_ground_truth if total_ground_truth > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'val_loss': val_loss / len(data_loader) if model.training else None,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': total_true_positives,
        'total_detections': total_detections,
        'total_ground_truth': total_ground_truth
    }
    
    return metrics


def main(args):
    """Main training function."""
    # Set up project paths
    project_root = Path(args.project_root)
    data_dir = project_root / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    output_dir = project_root / "outputs"
    model_dir = output_dir / "models"
    
    # Create necessary directories
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare data if needed
    if not (processed_data_dir / "train.csv").exists() or args.reprocess_data:
        logger.info("Preparing data...")
        data_dfs = prepare_data(project_root, save_eda_plots=True)
    else:
        logger.info("Using existing processed data...")
    
    # Create datasets
    train_dataset = HouseSparrowDataset(
        csv_file=processed_data_dir / "train.csv",
        img_dir=raw_data_dir / "images"
    )
    
    test_dataset = HouseSparrowDataset(
        csv_file=processed_data_dir / "test.csv",
        img_dir=raw_data_dir / "images"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
        
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize model - num_classes=2 (background and house sparrow)
    model = faster_rcnn_mob_model_for_n_classes(
        num_classes=2,
        print_head=True,
        trainable_backbone_layers=args.trainable_backbone_layers
    )
    model.to(device)
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"faster_rcnn_sparrow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "trainable_backbone_layers": args.trainable_backbone_layers,
            "device": device.type
        })
        
        # Store best model metrics
        best_f1 = 0.0
        best_model_path = None
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            logger.info(f"Epoch {epoch}/{args.epochs}")
            
            # Train one epoch
            train_metrics = train_one_epoch(model, optimizer, train_loader, device)
            
            # Update learning rate
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log training metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value, step=epoch)
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Time: {train_metrics['time_sec']:.2f}s, "
                        f"LR: {current_lr:.6f}")
            
            # Evaluate on test set
            eval_metrics = evaluate(model, test_loader, device)
            
            # Log evaluation metrics
            for metric_name, metric_value in eval_metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(f"eval_{metric_name}", metric_value, step=epoch)
            
            logger.info(f"Eval Precision: {eval_metrics['precision']:.4f}, "
                        f"Recall: {eval_metrics['recall']:.4f}, "
                        f"F1: {eval_metrics['f1_score']:.4f}")
            
            # Save model if it's the best one so far
            if eval_metrics['f1_score'] > best_f1:
                best_f1 = eval_metrics['f1_score']
                
                # Create timestamp for model name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"best_model_e{epoch}_f1_{best_f1:.4f}_{timestamp}.pth"
                best_model_path = model_dir / model_filename
                
                # Save the model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1_score': best_f1,
                    'precision': eval_metrics['precision'],
                    'recall': eval_metrics['recall']
                }, best_model_path)
                
                logger.info(f"Saved best model with F1={best_f1:.4f} to {best_model_path}")
                
                # Log the best model in MLflow
                mlflow.pytorch.log_model(model, "best_model")
        
        # Log final best metrics
        mlflow.log_metric("best_f1_score", best_f1)
        
        # Log the best model path as an artifact
        if best_model_path:
            mlflow.log_artifact(best_model_path)
        
        logger.info(f"Training completed. Best F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for House Sparrow Detection")
    
    # Data parameters
    parser.add_argument("--project_root", type=str, default=".",
                        help="Root directory of the project")
    parser.add_argument("--reprocess_data", action="store_true",
                        help="Force reprocess the data even if processed files exist")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay for regularization")
    parser.add_argument("--lr_step_size", type=int, default=3,
                        help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="Gamma for learning rate scheduler")
    parser.add_argument("--trainable_backbone_layers", type=int, default=3,
                        help="Number of trainable backbone layers")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    
    # MLflow parameters
    parser.add_argument("--mlflow_tracking_uri", type=str, default="mlruns",
                        help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="house_sparrow_detection",
                        help="MLflow experiment name")
    
    args = parser.parse_args()
    main(args)