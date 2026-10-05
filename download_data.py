"""
COS30082 Assignment 1
CUB-200 Dataset Downloader

This script downloads the CUB-200 dataset from our public
Hugging Face Dataset repository.

Dataset:
https://huggingface.co/datasets/print1Username/COS30082-Assignment-1

Downloaded files:
    - Train.zip
    - Test.zip
    - train.txt
    - test.txt

All dataset files are stored inside the "data" directory.

Before downloading, the script removes all existing files
and folders inside "data/", except for ".gitkeep".

This ensures that every download starts from a clean dataset
directory and prevents old or incomplete files from interfering
with the new download.

Expected directory structure:

    project/
    |
    +-- data/
    |   +-- .gitkeep
    |   +-- Train/
    |   +-- Test/
    |   +-- train.txt
    |   +-- test.txt
    |   +-- Train.zip
    |   +-- Test.zip
    |
    +-- download_data.py

The download progress shows:
    - Percentage
    - Downloaded size
    - Total size
    - Download speed
    - Estimated remaining time

No Hugging Face login or access token is required because
the dataset repository is public.
"""

import shutil
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import requests

# ============================================================
# Configuration
# ============================================================

# Hugging Face dataset repository.
REPO_ID = "print1Username/COS30082-Assignment-1"

# Hugging Face dataset branch.
BRANCH = "main"

# Local directory where the dataset will be stored.
DATA_DIR = Path("data")

# File that must always be preserved.
GITKEEP_FILE = DATA_DIR / ".gitkeep"

# Files that need to be downloaded.
DATA_FILES = [
	"Train.zip",
	"Test.zip",
	"train.txt",
	"test.txt",
]

# Size of each downloaded chunk.
# 1 MB provides a good balance between download speed
# and progress display frequency.
CHUNK_SIZE = 1024 * 1024

# Network timeout in seconds.
REQUEST_TIMEOUT = 60

# Number of retry attempts if a download fails.
MAX_RETRIES = 3


# ============================================================
# Formatting functions
# ============================================================

def format_size(size_bytes: int) -> str:
	"""
	Convert bytes into a human-readable size.

	Examples:
		1024 -> 1.00 KB
		1048576 -> 1.00 MB
		1073741824 -> 1.00 GB
	"""

	size = float(size_bytes)

	units = ["B", "KB", "MB", "GB", "TB"]

	for unit in units:
		if size < 1024:
			return f"{size:.2f} {unit}"

		size /= 1024

	return f"{size:.2f} PB"


def format_time(seconds: float) -> str:
	"""
	Convert seconds into a human-readable time format.

	Examples:
		65 -> 01:05
		3665 -> 01:01:05
	"""

	if seconds <= 0:
		return "00:00"

	seconds = int(seconds)

	hours, remainder = divmod(seconds, 3600)
	minutes, seconds = divmod(remainder, 60)

	if hours > 0:
		return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

	return f"{minutes:02d}:{seconds:02d}"


# ============================================================
# Clean data directory
# ============================================================

def clean_data_directory() -> None:
	"""
	Remove everything inside the data directory except
	for .gitkeep.

	This makes sure that every execution starts with a clean
	dataset directory.

	The data directory itself is NOT deleted.
	Only its contents are removed.

	.gitkeep is preserved so that Git can keep tracking the
	otherwise-empty data directory.
	"""

	print("\nPreparing data directory...")

	# Create data/ if it does not exist.
	DATA_DIR.mkdir(
		parents=True,
		exist_ok=True,
	)

	# Make sure .gitkeep exists.
	GITKEEP_FILE.touch(
		exist_ok=True
	)

	# Check every item inside data/.
	for item in DATA_DIR.iterdir():

		# Never delete .gitkeep.
		if item.name == ".gitkeep":
			continue

		try:

			# Remove directories recursively.
			if item.is_dir():
				shutil.rmtree(item)

			# Remove files.
			else:
				item.unlink()

			print(f"Removed: {item}")

		except PermissionError:
			raise PermissionError(
				f"Unable to remove '{item}'. "
				f"Please make sure the file is not being used "
				f"by another program."
			)

	print("Data directory cleaned successfully.")
	print("Preserved: data/.gitkeep")


# ============================================================
# Progress display
# ============================================================
def show_progress(filename, downloaded, total, speed, eta):
	"""
	Display download progress.

	If the server provides the total file size, show a progress bar
	and percentage.

	If the server does not provide the total file size, show the
	downloaded amount and speed instead.
	"""

	if total > 0:
		percent = downloaded / total * 100

		bar_length = 30
		filled = int(bar_length * downloaded / total)
		bar = "█" * filled + "░" * (bar_length - filled)

		print(
			f"\r{filename:<12} "
			f"[{bar}] "
			f"{percent:6.2f}% "
			f"{format_size(downloaded)} / {format_size(total)} "
			f"{format_size(speed)}/s "
			f"ETA {format_time(eta)}",
			end="",
			flush=True
		)

	else:
		# Some Hugging Face files may not provide Content-Length.
		# In this case, the total size is unknown, so percentage
		# and ETA cannot be calculated accurately.
		print(
			f"\r{filename:<12} "
			f"Downloaded {format_size(downloaded)} "
			f"{format_size(speed)}/s",
			end="",
			flush=True
		)


# ============================================================
# Download function
# ============================================================

def download_file(filename: str) -> Path:
	"""
	Download one file from Hugging Face.

	The file is downloaded in chunks so that real-time
	progress can be calculated and displayed.

	If the download fails, the incomplete file is removed
	and the download is retried.

	Returns:
		Path: Local path of the downloaded file.
	"""

	output_path = DATA_DIR / filename

	# Construct Hugging Face download URL.
	url = (
		f"https://huggingface.co/datasets/"
		f"{REPO_ID}/resolve/{BRANCH}/{filename}"
	)

	print(f"\nDownloading {filename}")

	for attempt in range(
		1,
		MAX_RETRIES + 1,
	):

		try:

			# Send HTTP request.
			response = requests.get(
				url,
				stream=True,
				timeout=REQUEST_TIMEOUT,
				allow_redirects=True,
			)

			# Raise an exception for HTTP errors.
			response.raise_for_status()

			# Read total file size.
			total_size = int(
				response.headers.get(
					"content-length",
					0,
				)
			)

			downloaded = 0
			start_time = time.time()

			# Open destination file.
			with open(
				output_path,
				"wb",
			) as file:

				# Download file chunk by chunk.
				for chunk in response.iter_content(
					chunk_size=CHUNK_SIZE
				):

					# Ignore empty chunks.
					if not chunk:
						continue

					# Save chunk to disk.
					file.write(chunk)

					# Update downloaded size.
					downloaded += len(chunk)

					# Calculate elapsed time.
					elapsed = (
						time.time()
						- start_time
					)

					# Calculate average speed.
					if elapsed > 0:
						speed = (
							downloaded
							/ elapsed
						)
					else:
						speed = 0

					# Update progress display.
					show_progress(
						filename,
						downloaded,
						total_size,
						speed,
						elapsed,
					)

			# Move to the next line after download.
			print()

			# Verify downloaded file size.
			actual_size = (
				output_path.stat().st_size
			)

			if (
				total_size > 0
				and actual_size != total_size
			):
				raise RuntimeError(
					f"File size mismatch. "
					f"Expected "
					f"{total_size} bytes, "
					f"received "
					f"{actual_size} bytes."
				)

			print(
				f"Completed: {filename} "
				f"({format_size(actual_size)})"
			)

			return output_path

		except Exception as error:

			print()

			print(
				f"Download failed for {filename} "
				f"(attempt "
				f"{attempt}/{MAX_RETRIES})."
			)

			print(
				f"Reason: {error}"
			)

			# Remove incomplete file.
			if output_path.exists():
				output_path.unlink()

			# Retry if attempts remain.
			if attempt < MAX_RETRIES:

				print(
					"Retrying in 2 seconds..."
				)

				time.sleep(2)

			else:

				raise RuntimeError(
					f"Failed to download "
					f"{filename} after "
					f"{MAX_RETRIES} attempts."
				)


# ============================================================
# Extraction function
# ============================================================

def extract_zip(
	zip_path: Path,
	output_dir: Path,
) -> None:
	"""
	Extract a ZIP file.

	A progress indicator is displayed during extraction.
	"""

	print(
		f"\nExtracting {zip_path.name}..."
	)

	with ZipFile(
		zip_path,
		"r",
	) as zip_file:
		members = zip_file.infolist()

		total_files = len(members)

		for index, member in enumerate(
			members,
			start=1,
		):
			zip_file.extract(
				member,
				DATA_DIR,
			)

			percentage = (
				index / total_files * 100
				if total_files > 0
				else 100
			)

			message = (
				f"\rExtracting "
				f"{zip_path.name}: "
				f"{percentage:6.2f}% "
				f"({index}/{total_files})"
			)

			sys.stdout.write(message)
			sys.stdout.flush()

	print()

	print(
		f"Finished extracting "
		f"{zip_path.name}."
	)


# ============================================================
# Dataset verification
# ============================================================

def verify_dataset() -> bool:
	"""
	Check whether all expected dataset files and directories
	exist after downloading and extraction.

	Returns:
		bool: True if the dataset appears complete.
	"""

	required_paths = [
		DATA_DIR / "Train",
		DATA_DIR / "Test",
		DATA_DIR / "train.txt",
		DATA_DIR / "test.txt",
	]

	print("\nChecking dataset...")

	all_exist = True

	for path in required_paths:

		if path.exists():

			print(
				f"[OK] {path}"
			)

		else:

			print(
				f"[MISSING] {path}"
			)

			all_exist = False

	# Also verify .gitkeep.
	if GITKEEP_FILE.exists():

		print(
			f"[OK] {GITKEEP_FILE}"
		)

	else:

		print(
			f"[MISSING] {GITKEEP_FILE}"
		)

		all_exist = False

	return all_exist


# ============================================================
# Main program
# ============================================================

def main() -> None:
	"""
	Main dataset download and extraction procedure.
	"""

	print("=" * 60)

	print(
		"COS30082 Assignment 1 - "
		"CUB-200 Dataset Downloader"
	)

	print("=" * 60)

	# --------------------------------------------------------
	# Step 1: Clean existing data
	# --------------------------------------------------------

	clean_data_directory()

	# --------------------------------------------------------
	# Step 2: Download dataset files
	# --------------------------------------------------------

	downloaded_files = {}

	for filename in DATA_FILES:
		downloaded_files[filename] = (
			download_file(filename)
		)

	# --------------------------------------------------------
	# Step 3: Extract training dataset
	# --------------------------------------------------------

	extract_zip(
		downloaded_files["Train.zip"],
		DATA_DIR / "Train",
	)

	# --------------------------------------------------------
	# Step 4: Extract testing dataset
	# --------------------------------------------------------

	extract_zip(
		downloaded_files["Test.zip"],
		DATA_DIR / "Test",
	)

	# --------------------------------------------------------
	# Step 5: Verify dataset
	# --------------------------------------------------------

	if verify_dataset():

		print(
			"\n" + "=" * 60
		)

		print(
			"Dataset download completed successfully!"
		)

		print(
			"=" * 60
		)

	else:

		print(
			"\n" + "=" * 60
		)

		print(
			"Dataset download completed with errors."
		)

		print(
			"Please check the missing files above."
		)

		print(
			"=" * 60
		)


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
	main()
