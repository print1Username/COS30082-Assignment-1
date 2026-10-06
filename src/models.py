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

The models can be created using:

    model = create_model("resnet18")
    model = create_model("resnet50")
    model = create_model("efficientnet_b0")

Training, loss functions, optimizers, and evaluation are handled
in other modules.
"""

import torch.nn as nn
from torchvision import models

# Number of classes in the CUB-200 dataset.
NUM_CLASSES = 200


def create_model(
	model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True, ):
	"""
	Create and return a classification model.

	Args:
		model_name (str):
			Name of the model to create.

			Supported values:
				- "resnet18"
				- "resnet50"
				- "efficientnet_b0"

		num_classes (int):
			Number of output classes.
			Default is 200 for CUB-200.

		pretrained (bool):
			Whether to use ImageNet pre-trained weights.
			Default is True.

	Returns:
		nn.Module:
			A PyTorch classification model configured for the
			specified number of classes.

	Raises:
		ValueError:
			If an unsupported model name is provided.
	"""

	# Convert the model name to lowercase so that inputs such as
	# "ResNet18" and "RESNET18" are also accepted.
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
			model = models.resnet18( weights = None )

		# Replace the original ImageNet classifier.
		# ImageNet has 1000 classes, while CUB-200 has 200 classes.
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
			model = models.resnet50( weights = None )

		# Replace the original ImageNet classifier with a
		# classifier suitable for the 200 CUB-200 classes.
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
			model = models.efficientnet_b0( weights = None )

		# EfficientNet stores its final classifier inside
		# model.classifier.
		model.classifier[1] = nn.Linear(
			model.classifier[1].in_features, num_classes
		)

	# ---------------------------------------------------------
	# Unsupported model
	# ---------------------------------------------------------
	else:
		raise ValueError(
			f"Unsupported model: '{model_name}'. "
			"Supported models are: "
			"'resnet18', 'resnet50', 'efficientnet_b0'."
		)

	return model


def freeze_backbone(model: nn.Module) -> None:
	"""
	Freeze all model parameters.

	This is useful for the first stage of transfer learning,
	where only the newly added classification layer is trained.

	Args:
		model (nn.Module):
			PyTorch model whose parameters should be frozen.
	"""

	for parameter in model.parameters():
		parameter.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
	"""
	Unfreeze all model parameters.

	This can be used during fine-tuning after the classifier
	has been trained.

	Args:
		model (nn.Module):
			PyTorch model whose parameters should be trainable.
	"""

	for parameter in model.parameters():
		parameter.requires_grad = True


def get_model_summary(model: nn.Module) -> dict:
	"""
	Return basic information about a model.

	Args:
		model (nn.Module):
			PyTorch model to inspect.

	Returns:
		dict:
			Dictionary containing total and trainable parameter counts.
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
	model = create_model( "resnet18" )

	print( model )
	print( "\nModel summary:" )
	print( get_model_summary( model ) )
