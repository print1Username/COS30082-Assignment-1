"""
models.py

Model definitions for the CUB-200 multi-class classification assignment.

This module provides a unified interface for creating different
pre-trained CNN models.

Supported models:
    1. ResNet18
    2. ResNet50
    3. EfficientNet-B0

All models use ImageNet pre-trained weights and are modified to
classify 200 bird species from the CUB-200 dataset.

The module also provides functions for:
    - Freezing the backbone while keeping the classifier trainable.
    - Unfreezing the entire model for fine-tuning.
    - Getting the number of model parameters.
"""

import torch.nn as nn
from torchvision import models

# CUB-200 contains 200 bird species.
NUM_CLASSES = 200


def create_model(
	model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True, ):
	"""
	Create a CNN model for CUB-200 classification.

	Args:
		model_name:
			Name of the model to create.
			Supported values:
				- "resnet18"
				- "resnet50"
				- "efficientnet_b0"

		num_classes:
			Number of output classes.

		pretrained:
			If True, load ImageNet-pretrained weights.

	Returns:
		A PyTorch model configured for the specified
		number of classes.
	"""

	model_name = model_name.lower()

	# ---------------------------------------------------------
	# ResNet18
	# ---------------------------------------------------------
	if model_name == "resnet18":

		if pretrained:
			model = models.resnet18(
				weights = models.ResNet18_Weights.DEFAULT
			)
		else:
			model = models.resnet18(
				weights = None
			)

		# Replace the original ImageNet classifier.
		# ImageNet has 1000 classes, while CUB-200 has 200.
		model.fc = nn.Linear(
			model.fc.in_features, num_classes
		)

	# ---------------------------------------------------------
	# ResNet50
	# ---------------------------------------------------------
	elif model_name == "resnet50":

		if pretrained:
			model = models.resnet50(
				weights = models.ResNet50_Weights.DEFAULT
			)
		else:
			model = models.resnet50(
				weights = None
			)

		# Replace the original ImageNet classifier.
		model.fc = nn.Linear(
			model.fc.in_features, num_classes
		)

	# ---------------------------------------------------------
	# EfficientNet-B0
	# ---------------------------------------------------------
	elif model_name == "efficientnet_b0":

		if pretrained:
			model = models.efficientnet_b0(
				weights = models.EfficientNet_B0_Weights.DEFAULT
			)
		else:
			model = models.efficientnet_b0(
				weights = None
			)

		# EfficientNet stores its classifier in model.classifier.
		# Replace the final Linear layer with a 200-class layer.
		model.classifier[1] = nn.Linear(
			model.classifier[1].in_features, num_classes
		)

	else:
		raise ValueError(
			f"Unsupported model: '{model_name}'. "
			"Supported models are: "
			"'resnet18', 'resnet50', 'efficientnet_b0'."
		)

	return model


def freeze_backbone(model: nn.Module) -> None:
	"""
	Freeze the feature-extraction backbone while keeping
	the final classification layer trainable.

	This is useful for the first stage of transfer learning.

	ResNet:
		- Freeze all parameters.
		- Unfreeze model.fc.

	EfficientNet:
		- Freeze all parameters.
		- Unfreeze model.classifier.

	Args:
		model:
			PyTorch model created by create_model().
	"""

	# First freeze every parameter in the model.
	for parameter in model.parameters():
		parameter.requires_grad = False

	# ---------------------------------------------------------
	# ResNet classifier
	# ---------------------------------------------------------
	if hasattr( model, "fc" ):

		for parameter in model.fc.parameters():
			parameter.requires_grad = True

	# ---------------------------------------------------------
	# EfficientNet classifier
	# ---------------------------------------------------------
	elif hasattr( model, "classifier" ):

		for parameter in model.classifier.parameters():
			parameter.requires_grad = True

	else:
		raise ValueError(
			"Unable to identify the classifier layer "
			"for this model."
		)


def unfreeze_model(model: nn.Module) -> None:
	"""
	Unfreeze all model parameters.

	This is used when performing full fine-tuning after
	the initial transfer-learning stage.

	Args:
		model:
			PyTorch model.
	"""

	for parameter in model.parameters():
		parameter.requires_grad = True


def get_model_summary(model: nn.Module) -> dict:
	"""
	Calculate the total and trainable number of parameters.

	Args:
		model:
			PyTorch model.

	Returns:
		Dictionary containing:
			- total_parameters
			- trainable_parameters
	"""

	total_parameters = sum(
		parameter.numel() for parameter in model.parameters()
	)

	trainable_parameters = sum(
		parameter.numel() for parameter in model.parameters() if parameter.requires_grad
	)

	return {
		"total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
	}


if __name__ == "__main__":

	print( "=" * 60 )
	print( "CUB-200 Model Test" )
	print( "=" * 60 )

	# ---------------------------------------------------------
	# Test ResNet18
	# ---------------------------------------------------------
	print( "\nCreating ResNet18..." )

	model = create_model(
		"resnet18", num_classes = NUM_CLASSES, pretrained = True
	)

	print( "\nModel summary before freezing:" )

	print(
		get_model_summary( model )
	)

	# ---------------------------------------------------------
	# Test freezing
	# ---------------------------------------------------------
	print( "\nFreezing ResNet18 backbone..." )

	freeze_backbone( model )

	print( "\nModel summary after freezing:" )

	print(
		get_model_summary( model )
	)

	# ---------------------------------------------------------
	# Test unfreezing
	# ---------------------------------------------------------
	print( "\nUnfreezing ResNet18..." )

	unfreeze_model( model )

	print( "\nModel summary after unfreezing:" )

	print(
		get_model_summary( model )
	)

	print( "\n" + "=" * 60 )
	print( "Model test completed successfully." )
	print( "=" * 60 )
