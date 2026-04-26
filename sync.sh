#!/bin/bash
# vibescent sync script

# 1. Capture current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "--- Syncing VibeScent ($BRANCH) ---"

# 2. Add all changes
git add .

# 3. Commit with timestamp
MSG="sync: $(date +'%Y-%m-%d %H:%M:%S')"
git commit -m "$MSG"

# 4. Push to remote
echo "Pushing to GitHub..."
git push origin "$BRANCH"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✓ Done! Changes are live on GitHub."
    echo "→ Now run the 'SYNC' cell in your Colab notebook."
    echo "------------------------------------------------"
else
    echo "❌ Push failed. Check your internet connection or GitHub permissions."
fi
