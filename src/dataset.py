"""
CUB-200 Dataset Loader

This module provides:
1. CUB200Dataset:
   - Reads image paths and labels from train.txt / test.txt.
   - Loads images using PIL.
   - Applies image transformations.

2. create_dataloaders():
   - Creates training, validation, and test DataLoaders.
   - Splits the official training set into training and validation subsets.
   - Keeps the official test set completely separate.

Dataset annotation format:
    image_name.jpg class_label

Example:
    001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg 1
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 200
IMAGE_SIZE = 224

# ImageNet normalization values.
# These are used because the models will use ImageNet-pretrained
# weights during transfer learning.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Transforms
# ============================================================

def get_train_transform():
	"""
	Create image transformations for training.

	Data augmentation is intentionally applied only to the
	training set to improve generalisation and reduce overfitting.
	"""

	return transforms.Compose(
		[
			transforms.RandomResizedCrop(
				IMAGE_SIZE, scale = (0.8, 1.0)
			),

			transforms.RandomHorizontalFlip(
				p = 0.5
			),

			transforms.RandomRotation(
				degrees = 15
			),

			transforms.ColorJitter(
				brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.05
			),

			transforms.ToTensor(),

			transforms.Normalize(
				mean = IMAGENET_MEAN, std = IMAGENET_STD
			)
		]
	)


def get_eval_transform():
	"""
	Create transformations for validation and testing.

	No random augmentation is used here so that validation and
	test results are consistent and reproducible.
	"""

	return transforms.Compose(
		[
			transforms.Resize(
				256
			),

			transforms.CenterCrop(
				IMAGE_SIZE
			),

			transforms.ToTensor(),

			transforms.Normalize(
				mean = IMAGENET_MEAN, std = IMAGENET_STD
			)
		]
	)


# ============================================================
# CUB-200 Dataset
# ============================================================

class CUB200Dataset( Dataset ):
	"""
	PyTorch Dataset for the CUB-200 dataset.

	The annotation file must contain:

		image_path class_label

	Example:

		001.Black_footed_Albatross/
		Black_Footed_Albatross_0001_796111.jpg 1
	"""

	def __init__(
		self, image_dir, annotation_file, transform = None
	):
		"""
		Parameters
		----------
		image_dir : str or Path
			Root directory containing the dataset images.

		annotation_file : str or Path
			Path to train.txt or test.txt.

		transform : torchvision.transforms.Compose, optional
			Image transformation pipeline.
		"""

		self.image_dir = Path( image_dir )
		self.annotation_file = Path( annotation_file )
		self.transform = transform

		self.samples = []

		self._load_annotations()

	def _load_annotations(self):
		"""
		Read image paths and labels from the annotation file.
		"""

		if not self.annotation_file.exists():
			raise FileNotFoundError(
				f"Annotation file not found: "
				f"{self.annotation_file}"
			)

		if not self.image_dir.exists():
			raise FileNotFoundError(
				f"Image directory not found: "
				f"{self.image_dir}"
			)

		with open(
			self.annotation_file, "r", encoding = "utf-8"
		) as file:

			for line_number, line in enumerate( file, start = 1 ):

				line = line.strip()

				# Skip empty lines.
				if not line:
					continue

				parts = line.split()

				if len( parts ) != 2:
					raise ValueError(
						f"Invalid annotation at line "
						f"{line_number}: {line}"
					)

				image_path, label = parts

				# The annotation files use zero-based class labels from 0 to 199.
				# This matches the class-index format expected by PyTorch
				# CrossEntropyLoss.
				label = int( label )

				if not 0 <= label < NUM_CLASSES:
					raise ValueError(
						f"Invalid label {label + 1} at "
						f"line {line_number}. "
						f"Expected labels from 1 to {NUM_CLASSES}."
					)

				full_image_path = (self.image_dir / image_path)

				self.samples.append(
					(
						full_image_path, label
					)
				)

		if len( self.samples ) == 0:
			raise ValueError(
				f"No samples found in {self.annotation_file}"
			)

	def __len__(self):
		"""
		Return the number of images in the dataset.
		"""

		return len( self.samples )

	def __getitem__(self, index):
		"""
		Load and return one image and its label.
		"""

		image_path, label = self.samples[index]

		if not image_path.exists():
			raise FileNotFoundError(
				f"Image not found: {image_path}"
			)

		# Convert to RGB because pretrained ImageNet models
		# expect three-channel RGB images.
		image = Image.open( image_path ).convert( "RGB" )

		if self.transform is not None:
			image = self.transform( image )

		return image, label


# ============================================================
# Dataset Splitting
# ============================================================

def create_train_validation_split(
	dataset, validation_size = 0.2, random_state = 42
):
	"""
	Split the training dataset into training and validation sets.

	Stratified splitting is used so that the distribution of
	the 200 bird classes remains approximately balanced.

	Parameters
	----------
	dataset : CUB200Dataset
		Full training dataset.

	validation_size : float
		Proportion used for validation.

	random_state : int
		Random seed for reproducibility.

	Returns
	-------
	train_indices : list
		Indices for the training subset.

	validation_indices : list
		Indices for the validation subset.
	"""

	labels = [label for _, label in dataset.samples]

	indices = list( range( len( dataset ) ) )

	train_indices, validation_indices = train_test_split(
		indices, test_size = validation_size, random_state = random_state, stratify = labels
	)

	return train_indices, validation_indices


# ============================================================
# DataLoaders
# ============================================================

def create_dataloaders(
	data_dir = "data", batch_size = 32, validation_size = 0.2, num_workers = 2, random_state = 42
):
	"""
	Create training, validation, and test DataLoaders.

	Expected directory structure:

		data/
		├── Train/
		│   ├── 001.Black_footed_Albatross/
		│   ├── 002.Laysan_Albatross/
		│   └── ...
		│
		├── Test/
		│   ├── 001.Black_footed_Albatross/
		│   ├── 002.Laysan_Albatross/
		│   └── ...
		│
		├── train.txt
		└── test.txt

	Returns
	-------
	train_loader
	validation_loader
	test_loader
	"""

	data_dir = Path( data_dir )

	train_dir = data_dir / "Train"
	test_dir = data_dir / "Test"

	train_annotation = data_dir / "train.txt"
	test_annotation = data_dir / "test.txt"

	# --------------------------------------------------------
	# Create the complete training dataset.
	# --------------------------------------------------------

	full_train_dataset = CUB200Dataset(
		image_dir = train_dir, annotation_file = train_annotation, transform = None
	)

	# --------------------------------------------------------
	# Create train/validation indices using stratified split.
	# --------------------------------------------------------

	train_indices, validation_indices = (create_train_validation_split(
		full_train_dataset, validation_size = validation_size, random_state = random_state
	))

	# --------------------------------------------------------
	# Create separate dataset objects so that training
	# augmentation is NOT applied to validation images.
	# --------------------------------------------------------

	train_dataset = CUB200Dataset(
		image_dir = train_dir, annotation_file = train_annotation, transform = get_train_transform()
	)

	validation_dataset = CUB200Dataset(
		image_dir = train_dir, annotation_file = train_annotation, transform = get_eval_transform()
	)

	# --------------------------------------------------------
	# Test dataset remains completely independent.
	# --------------------------------------------------------

	test_dataset = CUB200Dataset(
		image_dir = test_dir, annotation_file = test_annotation, transform = get_eval_transform()
	)

	# --------------------------------------------------------
	# Use the same indices generated from the original
	# training dataset.
	# --------------------------------------------------------

	train_subset = Subset(
		train_dataset, train_indices
	)

	validation_subset = Subset(
		validation_dataset, validation_indices
	)

	# --------------------------------------------------------
	# Create DataLoaders.
	# --------------------------------------------------------

	train_loader = DataLoader(
		train_subset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = torch.cuda.is_available()
	)

	validation_loader = DataLoader(
		validation_subset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = torch.cuda.is_available()
	)

	test_loader = DataLoader(
		test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = torch.cuda.is_available()
	)

	return (
		train_loader, validation_loader, test_loader
	)


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

	print( "=" * 60 )
	print( "CUB-200 Dataset Test" )
	print( "=" * 60 )

	train_loader, validation_loader, test_loader = (create_dataloaders(
		data_dir = "data", batch_size = 32, validation_size = 0.2, num_workers = 2, random_state = 42
	))

	print(
		f"Training batches: "
		f"{len( train_loader )}"
	)

	print(
		f"Validation batches: "
		f"{len( validation_loader )}"
	)

	print(
		f"Testing batches: "
		f"{len( test_loader )}"
	)

	# Load one batch to verify that everything works.
	images, labels = next( iter( train_loader ) )

	print(
		f"Image batch shape: "
		f"{images.shape}"
	)

	print(
		f"Label batch shape: "
		f"{labels.shape}"
	)

	print(
		f"Label range: "
		f"{labels.min().item()} - "
		f"{labels.max().item()}"
	)

	print( "=" * 60 )
	print( "Dataset test completed successfully." )
	print( "=" * 60 )

	all_labels = [label for _, label in train_loader.dataset.dataset.samples]

	print(
		f"Number of unique training classes: "
		f"{len( set( all_labels ) )}"
	)

	print(
		f"Minimum training label: "
		f"{min( all_labels )}"
	)

	print(
		f"Maximum training label: "
		f"{max( all_labels )}"
	)
