#!/bin/bash
# Script to download fall2025_deeplearning dataset using git-lfs
# and extract all zip files to a single directory

# Configuration
DATASET_REPO="https://huggingface.co/datasets/tsbpp/fall2025_deeplearning.git"
CLONE_DIR="fall2025_deeplearning"
OUTPUT_DIR="data/fall2025_dl/train"
TEMP_DIR="${CLONE_DIR}"

echo "=========================================="
echo "Downloading fall2025_deeplearning dataset"
echo "=========================================="
echo "Repository: $DATASET_REPO"
echo "Clone directory: $CLONE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is not installed"
    echo "Please install it with: module load git-lfs (or appropriate command for your system)"
    exit 1
fi

# Initialize git-lfs
git lfs install

# Clone the dataset repository
if [ -d "$CLONE_DIR" ]; then
    echo "Directory $CLONE_DIR already exists. Skipping clone."
    echo "If you want to re-download, please remove it first: rm -rf $CLONE_DIR"
else
    echo "Cloning repository (this may take a while for large datasets)..."
    git clone "$DATASET_REPO" "$CLONE_DIR"
    echo "✓ Repository cloned"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"

# Change to clone directory
cd "$CLONE_DIR"

# Find all zip files matching the pattern
ZIP_FILES=$(ls cc3m_96px_part*.zip 2>/dev/null || echo "")

if [ -z "$ZIP_FILES" ]; then
    echo "Warning: No zip files found matching pattern 'cc3m_96px_part*.zip'"
    echo "Listing all files in directory:"
    ls -lh
    exit 1
fi

ZIP_COUNT=$(echo "$ZIP_FILES" | wc -l)
echo ""
echo "Found $ZIP_COUNT zip file(s) to extract"
echo ""

# Extract all zip files to the output directory
# Using -o flag to overwrite without prompting, extracting to single directory
EXTRACTED_COUNT=0
FAILED_COUNT=0

echo "Extracting all zip files to: ../$OUTPUT_DIR"
echo ""

for f in cc3m_96px_part*.zip; do
    if [ -f "$f" ]; then
        echo "Extracting: $f"
        if unzip -o "$f" -d "../$OUTPUT_DIR" > /dev/null 2>&1; then
            EXTRACTED_COUNT=$((EXTRACTED_COUNT + 1))
            echo "  ✓ Extracted successfully"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "  ✗ Failed to extract"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Extraction Summary"
echo "=========================================="
echo "Total zip files: $ZIP_COUNT"
echo "Successfully extracted: $EXTRACTED_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Output directory: $(pwd)/../$OUTPUT_DIR"
echo ""

# Count extracted images
if [ -d "../$OUTPUT_DIR" ]; then
    IMAGE_COUNT=$(find "../$OUTPUT_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
    echo "Total images extracted: $IMAGE_COUNT"
fi

echo ""
echo "✓ Process completed!"

