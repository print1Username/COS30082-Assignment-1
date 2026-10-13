"""
evaluate.py

Evaluate a trained CUB-200 model on the official test set.

Reports:
    1. Top-1 accuracy:
        Percentage of all test images classified correctly.
    2. Average accuracy per class:
        Accuracy computed separately for each class, then averaged.
        This treats every bird species equally, regardless of how
        many test images it has.

The test set is used ONLY here, after training is finished.

Example:
    python src/evaluate.py --model resnet18
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import create_dataloaders
from models import create_model

# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 200

BATCH_SIZE = 32

DATA_DIR = "data"

CHECKPOINT_DIR = Path( "models" ) / "checkpoints"

RESULT_DIR = Path( "models" ) / "results"


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
	"""
	Select CUDA when available, otherwise CPU.
	"""

	if torch.cuda.is_available():
		return torch.device( "cuda" )

	return torch.device( "cpu" )


# ============================================================
# Model loading
# ============================================================

def load_best_model( model_name: str, device: torch.device ):
	"""
	Load the best checkpoint saved by train.py.

	Args:
		model_name:
			Model name, e.g. "resnet18".

		device:
			CUDA or CPU device.

	Returns:
		model:
			The model with the best weights, in eval mode.

		checkpoint:
			The full checkpoint dictionary (contains the
			validation accuracy saved during training).
	"""

	checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"

	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"Checkpoint not found: {checkpoint_path}. "
			f"Run train.py for this model first."
		)

	# pretrained = False: the weights come from the checkpoint,
	# so there is no need to download ImageNet weights again.
	model = create_model(
		model_name = model_name, num_classes = NUM_CLASSES, pretrained = False, )

	checkpoint = torch.load(
		checkpoint_path, map_location = device
	)

	model.load_state_dict(
		checkpoint[ "model_state_dict" ]
	)

	model = model.to( device )

	model.eval()

	return model, checkpoint


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model( model, data_loader, device: torch.device ):
	"""
	Compute Top-1 accuracy and average per-class accuracy.

	Counts are collected per class first, so the per-class
	average can be calculated exactly.

	Args:
		model:
			Trained PyTorch model in eval mode.

		data_loader:
			Test DataLoader.

		device:
			CUDA or CPU device.

	Returns:
		top1_accuracy:
			Overall accuracy (float between 0 and 1).

		average_per_class_accuracy:
			Mean of the per-class accuracies (float between 0 and 1).

		classes_evaluated:
			Number of classes that appear in the test set.
	"""

	model.eval()

	correct_per_class = torch.zeros( NUM_CLASSES, dtype = torch.long )

	total_per_class = torch.zeros( NUM_CLASSES, dtype = torch.long )

	progress_bar = tqdm(
		data_loader, desc = "Testing", leave = False, dynamic_ncols = True, )

	for images, labels in progress_bar:

		images = images.to(
			device, non_blocking = True
		)

		labels = labels.cpu()

		predictions = model( images ).argmax(
			dim = 1
		).cpu()

		# Count how many images each class has.
		total_per_class += torch.bincount(
			labels, minlength = NUM_CLASSES
		)

		# Count how many of them were predicted correctly.
		correct_mask = (predictions == labels)

		correct_per_class += torch.bincount(
			labels[ correct_mask ], minlength = NUM_CLASSES
		)

	# --------------------------------------------------------
	# Top-1 accuracy: total correct / total images
	# --------------------------------------------------------

	total_images = total_per_class.sum().item()

	top1_accuracy = (
		correct_per_class.sum().item() / total_images
	)

	# --------------------------------------------------------
	# Average per-class accuracy.
	# Only classes that actually appear in the test set are used,
	# to avoid dividing by zero.
	# --------------------------------------------------------

	present_classes = total_per_class > 0

	per_class_accuracy = (
		correct_per_class[ present_classes ].float()
		/ total_per_class[ present_classes ].float()
	)

	average_per_class_accuracy = (
		per_class_accuracy.mean().item()
	)

	classes_evaluated = int( present_classes.sum().item() )

	return top1_accuracy, average_per_class_accuracy, classes_evaluated


# ============================================================
# Save results
# ============================================================

def save_results( results: dict, model_name: str ) -> Path:
	"""
	Save the test results as a JSON file.
	"""

	RESULT_DIR.mkdir(
		parents = True, exist_ok = True
	)

	result_path = RESULT_DIR / f"{model_name}_test_results.json"

	with open(
		result_path, "w", encoding = "utf-8", ) as file:

		json.dump(
			results, file, indent = 4
		)

	return result_path


# ============================================================
# Main
# ============================================================

def evaluate( model_name: str ):
	"""
	Evaluate one trained model on the official test set.
	"""

	print( "\n" + "=" * 70 )
	print( f"Evaluating model: {model_name}" )
	print( "=" * 70 )

	device = get_device()

	print( f"Using device: {device}" )

	# --------------------------------------------------------
	# Test DataLoader.
	# create_dataloaders also returns train/validation loaders;
	# they are ignored here.
	# --------------------------------------------------------

	_, _, test_loader = create_dataloaders(
		data_dir = DATA_DIR, batch_size = BATCH_SIZE, )

	print( f"Test samples: {len( test_loader.dataset )}" )

	model, checkpoint = load_best_model(
		model_name, device
	)

	top1_accuracy, average_per_class_accuracy, classes_evaluated = (evaluate_model(
		model = model, data_loader = test_loader, device = device, ))

	if classes_evaluated < NUM_CLASSES:
		print(
			f"Warning: only {classes_evaluated} of {NUM_CLASSES} "
			f"classes appear in the test set."
		)

	results = {
		"model_name": model_name, "best_validation_accuracy": checkpoint[ "validation_accuracy" ], "test_top1_accuracy": top1_accuracy, "test_average_per_class_accuracy": average_per_class_accuracy, "test_samples": len( test_loader.dataset ), "classes_evaluated": classes_evaluated, }

	result_path = save_results(
		results, model_name
	)

	print( "\n" + "-" * 70 )
	print( f"Validation accuracy (best): {results[ 'best_validation_accuracy' ] * 100:.2f}%" )
	print( f"Test Top-1 accuracy:        {top1_accuracy * 100:.2f}%" )
	print( f"Test avg per-class acc:     {average_per_class_accuracy * 100:.2f}%" )
	print( "-" * 70 )
	print( f"Results saved to: {result_path}" )
	print( "=" * 70 )

	if classes_evaluated != NUM_CLASSES:
		raise ValueError(
			f"Expected {NUM_CLASSES} classes, "
			f"but found {classes_evaluated} in the test set."
		)


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments():

	parser = argparse.ArgumentParser(
		description = ("Evaluate a trained CUB-200 model on the test set.")
	)

	parser.add_argument(
		"--model", type = str, required = True, choices = [
			"resnet18", "resnet50", "efficientnet_b0",
		], help = ("Model architecture to evaluate."), )

	return parser.parse_args()


if __name__ == "__main__":

	args = parse_arguments()

	evaluate(
		model_name = args.model
	)