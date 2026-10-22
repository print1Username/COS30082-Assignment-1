"""
train.py

Training script for the CUB-200 multi-class classification assignment.

This script provides:
    1. GPU/CPU device selection.
    2. Reproducible training using a fixed random seed.
    3. Two-stage transfer learning:
        Stage 1:
            Train only the final classifier.
        Stage 2:
            Fine-tune the final feature stage and classifier.
    4. CrossEntropyLoss for 200-class classification.
    5. AdamW optimizer.
    6. ReduceLROnPlateau learning-rate scheduler.
    7. Early stopping to reduce overfitting.
    8. Mixed precision training when CUDA is available.
    9. Validation after every epoch.
    10. Saving the best validation model.
    11. Saving training history.

Supported models:
    - resnet18
    - resnet50
    - efficientnet_b0

Example:
    python src/train.py --model resnet18

    python src/train.py --model resnet50

    python src/train.py --model efficientnet_b0

The official test set is NOT used during training.
Final test evaluation is handled separately by evaluate.py.
"""

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import create_dataloaders
from models import (create_model, freeze_backbone, unfreeze_last_block, set_training_mode, get_model_summary, )

# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 200

BATCH_SIZE = 32

MAX_EPOCHS = 35

# Number of epochs used to train only the classifier.
WARMUP_EPOCHS = 5

# Learning rate for the classifier-only stage.
CLASSIFIER_LR = 1e-3

# Learning rate for final-stage fine-tuning.
FINETUNE_LR = 1e-5

# The new classifier can learn faster than the pretrained backbone.
FINETUNE_CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-3

# Early stopping patience.
# Training stops when validation loss does not improve
# for this many consecutive epochs.
EARLY_STOPPING_PATIENCE = 5

VALIDATION_SIZE = 0.2

RANDOM_SEED = 42

NUM_WORKERS = 2

DATA_DIR = "data"

CHECKPOINT_DIR = Path( "models" ) / "partial_finetune" / "checkpoints"

HISTORY_DIR = Path( "models" ) / "partial_finetune" / "history"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
	"""
	Set random seeds for reproducible experiments.

	Args:
		seed:
			Random seed value.
	"""

	random.seed( seed )

	np.random.seed( seed )

	torch.manual_seed( seed )

	if torch.cuda.is_available():
		torch.cuda.manual_seed( seed )
		torch.cuda.manual_seed_all( seed )

	# These settings improve reproducibility.
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
	"""
	Select CUDA when available, otherwise CPU.

	Returns:
		torch.device
	"""

	if torch.cuda.is_available():
		device = torch.device( "cuda" )
		print( f"Using GPU: {torch.cuda.get_device_name( 0 )}" )
	else:
		device = torch.device( "cpu" )
		print( "CUDA is not available. Using CPU." )

	return device


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, data_loader, criterion, optimizer, device, scaler, ):
	"""
	Train the model for one epoch.

	A tqdm progress bar is used to show the current training
	progress, loss, accuracy, and estimated remaining time.

	Args:
		model:
			PyTorch model.

		data_loader:
			Training DataLoader.

		criterion:
			Loss function.

		optimizer:
			Optimizer.

		device:
			CUDA or CPU device.

		scaler:
			AMP GradScaler.

	Returns:
		Average training loss and training accuracy.
	"""

	set_training_mode( model )

	running_loss = 0.0
	correct = 0
	total = 0

	progress_bar = tqdm(
		data_loader, desc = "Training", leave = False, dynamic_ncols = False, mininterval = 1.0, )

	for images, labels in progress_bar:

		images = images.to(
			device, non_blocking = True
		)

		labels = labels.to(
			device, non_blocking = True
		)

		optimizer.zero_grad(
			set_to_none = True
		)

		# Mixed precision is used when CUDA is available.
		with torch.amp.autocast(
			device_type = device.type, enabled = device.type == "cuda"
		):
			outputs = model( images )

			loss = criterion(
				outputs, labels
			)

		scaler.scale( loss ).backward()

		scaler.step( optimizer )

		scaler.update()

		running_loss += (loss.item() * images.size( 0 ))

		predictions = outputs.argmax(
			dim = 1
		)

		correct += (predictions == labels).sum().item()

		total += labels.size( 0 )

		# Calculate current running statistics.
		current_loss = (running_loss / total)

		current_accuracy = (correct / total)

		# Update progress bar.
		progress_bar.set_postfix(
			loss = f"{current_loss:.4f}", acc = f"{current_accuracy * 100:.2f}%", refresh = False
		)

	epoch_loss = running_loss / total

	epoch_accuracy = correct / total

	return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

def validate(model, data_loader, criterion, device, ):
	"""
	Evaluate the model on the validation set.

	A tqdm progress bar is used to show validation progress.

	The model is not updated during validation.

	Args:
		model:
			PyTorch model.

		data_loader:
			Validation DataLoader.

		criterion:
			Loss function.

		device:
			CUDA or CPU device.

	Returns:
		Average validation loss and validation accuracy.
	"""

	model.eval()

	running_loss = 0.0
	correct = 0
	total = 0

	progress_bar = tqdm(
		data_loader, desc = "Validation", leave = False, dynamic_ncols = False, mininterval = 1.0, )

	with torch.no_grad():

		for images, labels in progress_bar:

			images = images.to(
				device, non_blocking = True
			)

			labels = labels.to(
				device, non_blocking = True
			)

			with torch.amp.autocast(
				device_type = device.type, enabled = device.type == "cuda"
			):
				outputs = model( images )

				loss = criterion(
					outputs, labels
				)

			running_loss += (loss.item() * images.size( 0 ))

			predictions = outputs.argmax(
				dim = 1
			)

			correct += (predictions == labels).sum().item()

			total += labels.size( 0 )

			# Calculate current running statistics.
			current_loss = (running_loss / total)

			current_accuracy = (correct / total)

			# Update progress bar.
			progress_bar.set_postfix(
				loss = f"{current_loss:.4f}", acc = f"{current_accuracy * 100:.2f}%", refresh = False
			)

	epoch_loss = running_loss / total

	epoch_accuracy = correct / total

	return epoch_loss, epoch_accuracy


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, validation_accuracy, model_name, checkpoint_path, validation_loss, ):
	"""
	Save the best model checkpoint.

	The checkpoint contains:
		- Model weights
		- Optimizer state
		- Scheduler state
		- Epoch
		- Validation accuracy
		- Model name

	Args:
		model:
			PyTorch model.

		optimizer:
			Current optimizer.

		scheduler:
			Current scheduler.

		epoch:
			Current epoch.

		validation_accuracy:
			Validation accuracy.

		model_name:
			Model name.

		checkpoint_path:
			Output checkpoint path.

		validation_loss:
			Validation loss at the saved epoch.
	"""

	checkpoint = {
		"epoch": epoch, "model_name": model_name, "validation_accuracy": validation_accuracy, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
		"validation_loss": validation_loss,
		"training_config": {
			"finetune_scope": "last_feature_stage_and_classifier",
			"classifier_lr": CLASSIFIER_LR, "backbone_lr": FINETUNE_LR,
			"finetune_classifier_lr": FINETUNE_CLASSIFIER_LR, "weight_decay": WEIGHT_DECAY,
			"label_smoothing": 0.1, "warmup_epochs": WARMUP_EPOCHS,
			"max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
			"validation_size": VALIDATION_SIZE, "seed": RANDOM_SEED,
			"early_stopping_metric": "validation_loss", "patience": EARLY_STOPPING_PATIENCE,
		},
	}

	torch.save(
		checkpoint, checkpoint_path
	)


# ============================================================
# Main Training Function
# ============================================================

def train_model(model_name: str):
	"""
	Train one CUB-200 model.

	Args:
		model_name:
			Model name:
				- resnet18
				- resnet50
				- efficientnet_b0
	"""

	print( "\n" + "=" * 70 )
	print( f"Training model: {model_name}" )
	print( "=" * 70 )

	# --------------------------------------------------------
	# Reproducibility
	# --------------------------------------------------------

	set_seed( RANDOM_SEED )

	# --------------------------------------------------------
	# Device
	# --------------------------------------------------------

	device = get_device()

	# --------------------------------------------------------
	# Create DataLoaders
	# --------------------------------------------------------

	print( "\nCreating DataLoaders..." )

	train_loader, validation_loader, _ = (create_dataloaders(
		data_dir = DATA_DIR, batch_size = BATCH_SIZE, validation_size = VALIDATION_SIZE, num_workers = NUM_WORKERS, random_state = RANDOM_SEED, ))

	print(
		f"Training samples: "
		f"{len( train_loader.dataset )}"
	)

	print(
		f"Validation samples: "
		f"{len( validation_loader.dataset )}"
	)

	# --------------------------------------------------------
	# Create model
	# --------------------------------------------------------

	print( "\nCreating model..." )

	model = create_model(
		model_name = model_name, num_classes = NUM_CLASSES, pretrained = True, )

	model = model.to( device )

	print( "\nModel summary:" )

	print(
		get_model_summary( model )
	)

	# --------------------------------------------------------
	# Loss function
	# --------------------------------------------------------

	criterion = nn.CrossEntropyLoss( label_smoothing = 0.1 )

	# --------------------------------------------------------
	# Stage 1:
	# Train classifier only
	# --------------------------------------------------------

	print( "\n" + "-" * 70 )
	print( "Stage 1: Training classifier only" )
	print( "-" * 70 )

	freeze_backbone( model )

	print(
		"Trainable parameters after freezing:"
	)

	print(
		get_model_summary( model )
	)

	optimizer = AdamW(
		filter(
			lambda parameter: parameter.requires_grad, model.parameters()
		), lr = CLASSIFIER_LR, weight_decay = WEIGHT_DECAY, )

	scheduler = ReduceLROnPlateau(
		optimizer, mode = "min", factor = 0.5, patience = 2, )

	# --------------------------------------------------------
	# Mixed precision scaler
	# --------------------------------------------------------

	scaler = torch.amp.GradScaler(
		device = device.type, enabled = device.type == "cuda"
	)

	# --------------------------------------------------------
	# Checkpoint paths
	# --------------------------------------------------------

	CHECKPOINT_DIR.mkdir(
		parents = True, exist_ok = True
	)

	HISTORY_DIR.mkdir(
		parents = True, exist_ok = True
	)

	checkpoint_path = (CHECKPOINT_DIR / f"{model_name}_best.pth")

	history_path = (HISTORY_DIR / f"{model_name}_history.json")

	# --------------------------------------------------------
	# Training history
	# --------------------------------------------------------

	history = {
		"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": [], "learning_rate": [],
		"classifier_learning_rate": [],
	}

	best_validation_accuracy = -1.0
	best_validation_loss = float( "inf" )

	epochs_without_improvement = 0

	# --------------------------------------------------------
	# Training loop
	# --------------------------------------------------------

	for epoch in range( 1, MAX_EPOCHS + 1 ):

		# ----------------------------------------------------
		# Switch to final-stage fine-tuning after warm-up.
		# ----------------------------------------------------

		if epoch == WARMUP_EPOCHS + 1:

			print( "\n" + "-" * 70 )
			print( "Stage 2: Fine-tuning final feature stage and classifier" )
			print( "-" * 70 )

			unfreeze_last_block( model )

			print(
				"Trainable parameters after unfreezing:"
			)

			print(
				get_model_summary( model )
			)

			# Protect pretrained features with a lower backbone learning rate.
			classifier = model.fc if hasattr( model, "fc" ) else model.classifier
			classifier_ids = { id( parameter ) for parameter in classifier.parameters() }
			optimizer = AdamW(
				[
					{ "params": [p for p in model.parameters() if p.requires_grad and id( p ) not in classifier_ids], "lr": FINETUNE_LR },
					{ "params": classifier.parameters(), "lr": FINETUNE_CLASSIFIER_LR },
				], weight_decay = WEIGHT_DECAY, )

			scheduler = ReduceLROnPlateau(
				optimizer, mode = "min", factor = 0.5, patience = 2, )
			# Give fine-tuning its own loss-based patience window.
			best_validation_loss = float( "inf" )
			epochs_without_improvement = 0

		# ----------------------------------------------------
		# Train
		# ----------------------------------------------------

		train_loss, train_accuracy = (train_one_epoch(
			model = model, data_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device, scaler = scaler, ))

		# ----------------------------------------------------
		# Validation
		# ----------------------------------------------------

		validation_loss, validation_accuracy = (validate(
			model = model, data_loader = validation_loader, criterion = criterion, device = device, ))

		# ----------------------------------------------------
		# Scheduler
		# ----------------------------------------------------

		scheduler.step(
			validation_loss
		)

		current_learning_rate = (optimizer.param_groups[0]["lr"])

		# ----------------------------------------------------
		# Save history
		# ----------------------------------------------------

		history["train_loss"].append(
			train_loss
		)

		history["train_accuracy"].append(
			train_accuracy
		)

		history["validation_loss"].append(
			validation_loss
		)

		history["validation_accuracy"].append(
			validation_accuracy
		)

		history["learning_rate"].append(
			current_learning_rate
		)
		history["classifier_learning_rate"].append( optimizer.param_groups[-1]["lr"] )

		# ----------------------------------------------------
		# Print epoch results
		# ----------------------------------------------------

		print(
			f"\nEpoch [{epoch:02d}/{MAX_EPOCHS}]"
		)

		print(
			f"Train Loss: "
			f"{train_loss:.4f}"
		)

		print(
			f"Train Accuracy: "
			f"{train_accuracy * 100:.2f}%"
		)

		print(
			f"Validation Loss: "
			f"{validation_loss:.4f}"
		)

		print(
			f"Validation Accuracy: "
			f"{validation_accuracy * 100:.2f}%"
		)

		print(
			f"Learning Rate: "
			f"{current_learning_rate:.6f}"
		)

		# ----------------------------------------------------
		# Check for improvement
		# ----------------------------------------------------

		if validation_accuracy > best_validation_accuracy:

			best_validation_accuracy = (validation_accuracy)

			save_checkpoint(
				model = model, optimizer = optimizer, scheduler = scheduler, epoch = epoch, validation_accuracy = (validation_accuracy), model_name = model_name, checkpoint_path = checkpoint_path, validation_loss = validation_loss, )

			print(
				"✓ Best model saved."
			)

		# Save by accuracy for deployment, but stop by loss to detect overconfidence.
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			epochs_without_improvement = 0
		else:

			epochs_without_improvement += 1

			print(
				f"No validation loss improvement for "
				f"{epochs_without_improvement} epoch(s)."
			)

		# Persist completed epochs so interrupted local runs retain their history.
		with open( history_path, "w", encoding = "utf-8" ) as file:
			json.dump( history, file, indent = 4 )

		# ----------------------------------------------------
		# Early stopping
		# ----------------------------------------------------

		if (epochs_without_improvement >= EARLY_STOPPING_PATIENCE):

			print(
				"\nEarly stopping triggered."
			)

			break

	# --------------------------------------------------------
	# Save training history
	# --------------------------------------------------------

	with open(
		history_path, "w", encoding = "utf-8", ) as file:

		json.dump(
			history, file, indent = 4, )

	# --------------------------------------------------------
	# Final information
	# --------------------------------------------------------

	print( "\n" + "=" * 70 )

	print(
		f"Training completed: {model_name}"
	)

	print(
		f"Best validation accuracy: "
		f"{best_validation_accuracy * 100:.2f}%"
	)

	print(
		f"Best checkpoint: "
		f"{checkpoint_path}"
	)

	print(
		f"Training history: "
		f"{history_path}"
	)

	print( "=" * 70 )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments():
	"""
	Parse command-line arguments.
	"""

	parser = argparse.ArgumentParser(
		description = ("Train a CUB-200 image classification model.")
	)

	parser.add_argument(
		"--model", type = str, required = True, choices = [
			"resnet18", "resnet50", "efficientnet_b0",
		], help = ("Model architecture to train."), )

	return parser.parse_args()


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

	args = parse_arguments()

	train_model(
		model_name = args.model
	)
