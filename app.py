"""
app.py

COS30082 Assignment 1
CUB-200 Bird Species Classification Web Application

Features:
1. Print PyTorch and CUDA environment information.
2. Select the trained checkpoint with the highest validation accuracy.
3. Load the mapping between class indices and bird species.
4. Predict the top-5 bird species from an uploaded image.
5. Run on either CPU or CUDA.
"""

import os
import json
import math
from pathlib import Path

import gradio as gr
import torch
from torchvision import transforms
from huggingface_hub import HfApi, hf_hub_download
from src.models import create_model

# ============================================================
# 1. Environment Version Check
# Keep this section at the beginning, as requested.
# ============================================================

print( "PyTorch version:", torch.__version__ )
print( "CUDA available:", torch.cuda.is_available() )
print( "CUDA version:", torch.version.cuda )
print( "GPU count:", torch.cuda.device_count() )

if torch.cuda.is_available():
	print( "GPU:", torch.cuda.get_device_name( 0 ) )

print()
print()

# ============================================================
# 2. Application Configuration
# ============================================================

# Hugging Face Model repository containing the trained weights
MODEL_REPO_ID = ("print1Username/COS30082-Assignment-1-model")

# Expected class mapping filename.
MAPPING_FILENAME = "class_mapping.json"

NUM_CLASSES = 200
IMAGE_SIZE = 224
TOP_K = 5

# Use CUDA if available; otherwise, use CPU.
DEVICE = torch.device(
	"cuda" if torch.cuda.is_available() else "cpu"
)

# The project root is the directory containing app.py.
APP_DIR = Path( __file__ ).resolve().parent

# Prefer local checkpoints; use Hugging Face when no local checkpoints exist.
LOCAL_CHECKPOINT_DIR = APP_DIR / "models" / "checkpoints"
SUPPORTED_MODELS = ("resnet18", "resnet50", "efficientnet_b0")

# Image preprocessing must match the evaluation pipeline
# used during training.
IMAGE_TRANSFORM = transforms.Compose(
	[
		transforms.Resize( 256 ), transforms.CenterCrop( IMAGE_SIZE ), transforms.ToTensor(), transforms.Normalize(
		mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], ),
	]
)


# ============================================================
# 3. Find and Download the Model Checkpoint
# ============================================================

def select_best_checkpoint(checkpoint_paths):
	"""Rank the checkpoints themselves, never stale history or test-result JSON.

	Only one checkpoint is read at a time on CPU. Ties use filename order,
	so selection is reproducible and does not favour a particular architecture.
	"""
	best_path = None
	best_accuracy = -1.0
	for path in sorted( checkpoint_paths, key = str ):
		checkpoint = torch.load( path, map_location = "cpu", weights_only = True )
		model_name = checkpoint.get( "model_name" )
		accuracy = checkpoint.get( "validation_accuracy" )
		if model_name not in SUPPORTED_MODELS:
			raise ValueError( f"Unsupported or missing model_name in checkpoint: {path}" )
		if not isinstance( accuracy, (int, float) ) or not math.isfinite( accuracy ) or not 0 <= accuracy <= 1:
			raise ValueError( f"Checkpoint must record validation_accuracy between 0 and 1: {path}" )
		print( f"Candidate: {model_name}, validation accuracy: {accuracy:.2%}" )
		if accuracy > best_accuracy:
			best_path, best_accuracy = path, accuracy
		del checkpoint
	if best_path is None:
		raise FileNotFoundError( "No trained model checkpoints were found." )
	return str( best_path )


def get_checkpoint_path():
	"""Select the best available local model, or rank repository models on fallback."""
	local_paths = [p for p in LOCAL_CHECKPOINT_DIR.glob( "*" ) if p.suffix.lower() in (".pth", ".pt", ".bin")]
	if local_paths:
		return select_best_checkpoint( local_paths )

	repo_files = HfApi().list_repo_files( repo_id = MODEL_REPO_ID, repo_type = "model" )
	# Backups in subfolders are not deployment candidates.
	filenames = sorted( name for name in repo_files if "/" not in name and name.lower().endswith( (".pth", ".pt", ".bin") ) )
	paths = (hf_hub_download( repo_id = MODEL_REPO_ID, filename = name, repo_type = "model" ) for name in filenames)
	return select_best_checkpoint( paths )


# ============================================================
# 4. Load the Class Mapping
# ============================================================

def load_class_mapping():
	"""
	Load class_mapping.json.

	The expected format is a JSON dictionary such as:
		{
			"0": "Black footed Albatross",
			"1": "Laysan Albatross"
		}

	The numeric keys must match the labels used during training.
	"""

	# First, check whether the mapping is stored beside app.py.
	local_mapping = APP_DIR / MAPPING_FILENAME

	if local_mapping.is_file():
		mapping_path = local_mapping
	else:
		# Otherwise, retrieve it from the Model repository.
		mapping_path = Path(
			hf_hub_download(
				repo_id = MODEL_REPO_ID, filename = MAPPING_FILENAME, repo_type = "model", )
		)

	with open(
		mapping_path, "r", encoding = "utf-8", ) as file:
		raw_mapping = json.load( file )

	# Convert JSON keys to integers for direct indexing.
	mapping = { int( class_index ): str( species_name ) for class_index, species_name in raw_mapping.items() }

	expected_indices = set( range( NUM_CLASSES ) )

	if set( mapping.keys() ) != expected_indices:
		raise ValueError(
			"class_mapping.json must contain every class "
			f"index from 0 to {NUM_CLASSES - 1}."
		)

	if any( not name.strip() for name in mapping.values() ):
		raise ValueError(
			"class_mapping.json contains an empty species name."
		)

	print(
		f"Loaded {len( mapping )} bird species from:", mapping_path, )

	return mapping


# ============================================================
# 5. Build the Model Saved in the Checkpoint
# ============================================================

def create_inference_model():
	"""
	Recreate the architecture recorded in the training checkpoint,
	then load its trained weights.

	The model name must be one supported by src.models.create_model().
	"""

	checkpoint_path = get_checkpoint_path()

	# The checkpoint may contain either a raw state dictionary
	# or a training dictionary containing model_state_dict.
	try:
		checkpoint = torch.load(
			checkpoint_path, map_location = "cpu", weights_only = True, )
	except TypeError:
		# Compatibility with older PyTorch versions.
		checkpoint = torch.load(
			checkpoint_path, map_location = "cpu", )

	if isinstance( checkpoint, dict ):
		if "model_state_dict" in checkpoint:
			state_dict = checkpoint["model_state_dict"]
		elif "state_dict" in checkpoint:
			state_dict = checkpoint["state_dict"]
		else:
			state_dict = checkpoint
	else:
		raise ValueError(
			"The checkpoint does not contain a valid "
			"PyTorch state dictionary."
		)

	model_name = checkpoint.get( "model_name" )

	if not isinstance( model_name, str ):
		raise ValueError(
			"The checkpoint must contain a string 'model_name' "
			"so the inference architecture can be recreated."
		)

	model = create_model(
		model_name = model_name, num_classes = NUM_CLASSES, pretrained = False, )

	# Handle checkpoints saved from DataParallel models.
	state_dict = { key.removeprefix( "module." ): value for key, value in state_dict.items() }

	# Strict loading catches architecture or weight mismatches.
	model.load_state_dict( state_dict, strict = True )

	model.to( DEVICE )
	model.eval()

	print( "Model loaded successfully." )
	print( "Model architecture:", model_name )
	print( "Inference device:", DEVICE )

	return model, model_name, checkpoint["validation_accuracy"]


# ============================================================
# 6. Initialise the Model and Species Names
# ============================================================

# Load once when the application starts, not for every image.
CLASS_MAPPING = load_class_mapping()
MODEL, MODEL_NAME, VALIDATION_ACCURACY = create_inference_model()
MODEL_LABEL = {"resnet18": "ResNet18", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"}[MODEL_NAME]


# ============================================================
# 7. Predict the Top-5 Bird Species
# ============================================================

def predict_bird(image):
	"""
	Accept an uploaded PIL image and return the top-5
	predicted bird species with their confidence scores.

	Gradio's Label component expects a dictionary mapping
	class names to confidence scores.
	"""

	if image is None:
		return { }

	# Ensure the uploaded image has three RGB channels.
	image = image.convert( "RGB" )

	# Convert the image into a normalised PyTorch tensor.
	image_tensor = IMAGE_TRANSFORM( image ).unsqueeze( 0 )
	image_tensor = image_tensor.to( DEVICE )

	# Disable gradient calculations for inference.
	with torch.inference_mode():
		logits = MODEL( image_tensor )

		# Convert the 200 output logits into probabilities.
		probabilities = torch.softmax( logits, dim = 1 )[0]

		# Retrieve the top-5 predictions.
		top_probabilities, top_indices = torch.topk(
			probabilities, k = TOP_K, )

	# Format the result for Gradio.
	predictions = { }

	for probability, class_index in zip(
		top_probabilities.cpu().tolist(), top_indices.cpu().tolist(), ):
		species_name = CLASS_MAPPING[class_index]

		predictions[species_name] = float( probability )

	return predictions


# ============================================================
# 8. Gradio Web Interface
# ============================================================

with gr.Blocks(
	title = "CUB-200 Bird Species Classifier", ) as demo:

	gr.Markdown(
		f"""
		# CUB-200 Bird Species Classifier

		Upload a bird image to classify it into one of
		the 200 bird species in the CUB-200 dataset.

		The application uses a fine-tuned {MODEL_LABEL} model developed for COS30082 Assignment 1.
		It is selected automatically by validation accuracy among available checkpoints.
		"""
	)

	with gr.Row():
		with gr.Column():
			input_image = gr.Image(
				type = "pil", sources = ["upload", "clipboard"], label = "Upload a Bird Image", )

			predict_button = gr.Button(
				"Classify Bird", variant = "primary", )

		with gr.Column():
			output_label = gr.Label(
				num_top_classes = TOP_K, label = "Top-5 Predictions", )

	predict_button.click(
		fn = predict_bird, inputs = input_image, outputs = output_label, )

	gr.Markdown(
		f"""
		---
		**Model:** {MODEL_LABEL}

		**Validation accuracy:** {VALIDATION_ACCURACY:.2%}

		**Dataset:** CUB-200, 200 bird species

		**Input size:** 224 × 224 pixels

		**Inference device:** {DEVICE}
		"""
	)

# ============================================================
# 9. Launch the Application
# ============================================================

if __name__ == "__main__":
	demo.launch(
		server_name = "0.0.0.0", server_port = int( os.environ.get( "PORT", 7860 ) ), )
