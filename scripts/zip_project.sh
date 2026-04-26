#!/bin/bash
# Zips the project for upload to Google Drive, ignoring heavy/unneeded directories.

echo "Creating vibescent.zip..."
cd ..

# Ensure we remove any existing zip
rm -f vibescent.zip

# Zip the directory, excluding heavy and temporary folders
zip -rq vibescent.zip vibescent \
    -x "vibescent/.git/*" \
    -x "vibescent/.venv/*" \
    -x "vibescent/node_modules/*" \
    -x "vibescent/artifacts/*" \
    -x "vibescent/checkpoints/*" \
    -x "vibescent/__pycache__/*" \
    -x "vibescent/.next/*" \
    -x "vibescent/embeddings/*" \
    -x "vibescent/notebooks/.ipynb_checkpoints/*"

# Move the zip back inside the project directory (or keep it outside)
mv vibescent.zip vibescent/vibescent.zip

echo "Created vibescent/vibescent.zip successfully!"
echo "Upload this file to the root of your Google Drive."