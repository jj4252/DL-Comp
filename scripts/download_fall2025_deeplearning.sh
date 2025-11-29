#!/bin/bash
# Script to download fall2025_deeplearning zip files from HuggingFace using curl
# and extract them to a single directory

# Configuration
DATASET_NAME="tsbpp/fall2025_deeplearning"
BASE_URL="https://huggingface.co/datasets/${DATASET_NAME}/resolve/main"
ZIP_DIR="fall2025_deeplearning"
OUTPUT_DIR="data/fall2025_dl/train"

# Try to auto-detect zip files from HuggingFace, or use manual list
# First, try to get file list from HuggingFace API (may not work for git-lfs files)
# Otherwise, you'll need to manually specify the filenames

# Option 1: Try to discover files (may not work for git-lfs)
# Option 2: Manual list - update this with actual filenames from:
# https://huggingface.co/datasets/tsbpp/fall2025_deeplearning/tree/main

# For now, using a pattern-based approach - will try common patterns
# You may need to check the HuggingFace repo and update the list
ZIP_PATTERNS=(
    "cc3m_96px_part01.zip"
    "cc3m_96px_part02.zip"
    "cc3m_96px_part03.zip"
    "cc3m_96px_part04.zip"
    "cc3m_96px_part05.zip"
    "cc3m_96px_part06.zip"
    "cc3m_96px_part07.zip"
    "cc3m_96px_part08.zip"
    "cc3m_96px_part09.zip"
    "cc3m_96px_part10.zip"
    "cc3m_96px_part11.zip"
    "cc3m_96px_part12.zip"
    "cc3m_96px_part13.zip"
    "cc3m_96px_part14.zip"
    "cc3m_96px_part15.zip"
    "cc3m_96px_part16.zip"
    "cc3m_96px_part17.zip"
    "cc3m_96px_part18.zip"
    "cc3m_96px_part19.zip"
    "cc3m_96px_part20.zip"
)

echo "=========================================="
echo "Downloading fall2025_deeplearning dataset"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Base URL: $BASE_URL"
echo "Zip directory: $ZIP_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Verify dataset exists using HuggingFace datasets-server API
echo "Verifying dataset access..."
if curl -s -f "https://datasets-server.huggingface.co/splits?dataset=tsbpp%2Ffall2025_deeplearning" > /dev/null 2>&1; then
    echo "✓ Dataset is accessible via HuggingFace"
else
    echo "⚠ Warning: Could not verify dataset access (may still work for direct file downloads)"
fi
echo ""

# Create zip directory
mkdir -p "$ZIP_DIR"
cd "$ZIP_DIR"

# Download zip files
DOWNLOADED_COUNT=0
FAILED_COUNT=0

echo "Downloading zip files..."
echo ""

for zip_file in "${ZIP_PATTERNS[@]}"; do
    if [ -f "$zip_file" ]; then
        echo "Skipping $zip_file (already exists)"
        DOWNLOADED_COUNT=$((DOWNLOADED_COUNT + 1))
    else
        echo "Downloading: $zip_file"
        URL="${BASE_URL}/${zip_file}"
        
        # Try downloading with curl, show progress for large files
        if curl -L -f --progress-bar -o "$zip_file" "$URL" 2>&1; then
            # Verify the file is not empty and is a valid zip
            if [ -s "$zip_file" ] && unzip -tq "$zip_file" > /dev/null 2>&1; then
                DOWNLOADED_COUNT=$((DOWNLOADED_COUNT + 1))
                FILE_SIZE=$(du -h "$zip_file" | cut -f1)
                echo "  ✓ Downloaded successfully ($FILE_SIZE)"
            else
                FAILED_COUNT=$((FAILED_COUNT + 1))
                echo "  ✗ Downloaded file is invalid or corrupted"
                rm -f "$zip_file"
            fi
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "  ✗ Failed to download (file may not exist, require git-lfs, or URL incorrect)"
            # Remove partial download if it exists
            rm -f "$zip_file"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "Total files to try: ${#ZIP_PATTERNS[@]}"
echo "Successfully downloaded: $DOWNLOADED_COUNT"
echo "Failed: $FAILED_COUNT"
echo ""

# Check if we have any zip files
ACTUAL_ZIP_FILES=$(ls cc3m_96px_part*.zip 2>/dev/null | wc -l)

if [ "$ACTUAL_ZIP_FILES" -eq 0 ]; then
    echo "Warning: No zip files found. This could mean:"
    echo "  1. Files are stored with git-lfs and require git-lfs to download"
    echo "  2. The filenames in the script don't match the HuggingFace repository"
    echo "  3. Direct curl downloads are not supported for this dataset"
    echo ""
    echo "Solutions:"
    echo "  - Check available files at: https://huggingface.co/datasets/${DATASET_NAME}/tree/main"
    echo "  - Update ZIP_PATTERNS array in the script with correct filenames"
    echo "  - Use git-lfs to clone the repository, or"
    echo "  - Use the unzip script (unzip_fall2025_deeplearning.sh) after manual transfer"
    echo ""
    echo "You can also try querying the dataset info:"
    echo "  curl 'https://datasets-server.huggingface.co/splits?dataset=tsbpp%2Ffall2025_deeplearning'"
    exit 1
fi

echo "Found $ACTUAL_ZIP_FILES zip file(s) ready for extraction"
echo ""

# Go back to project root
cd ..

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"

# Change to zip directory for extraction
cd "$ZIP_DIR"

# Extract all zip files to the output directory
EXTRACTED_COUNT=0
EXTRACT_FAILED_COUNT=0

echo "Extracting all zip files to: ../$OUTPUT_DIR"
echo ""

for f in cc3m_96px_part*.zip; do
    if [ -f "$f" ]; then
        echo "Extracting: $f"
        if unzip -o "$f" -d "../$OUTPUT_DIR" > /dev/null 2>&1; then
            EXTRACTED_COUNT=$((EXTRACTED_COUNT + 1))
            echo "  ✓ Extracted successfully"
        else
            EXTRACT_FAILED_COUNT=$((EXTRACT_FAILED_COUNT + 1))
            echo "  ✗ Failed to extract"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Extraction Summary"
echo "=========================================="
echo "Total zip files: $ACTUAL_ZIP_FILES"
echo "Successfully extracted: $EXTRACTED_COUNT"
echo "Failed: $EXTRACT_FAILED_COUNT"
echo "Output directory: $(pwd)/../$OUTPUT_DIR"
echo ""

# Count extracted images
if [ -d "../$OUTPUT_DIR" ]; then
    IMAGE_COUNT=$(find "../$OUTPUT_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
    echo "Total images extracted: $IMAGE_COUNT"
fi

echo ""
echo "✓ Process completed!"

