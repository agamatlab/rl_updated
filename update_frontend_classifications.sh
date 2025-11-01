#!/bin/bash
#
# Script to update the frontend with latest classification data
# Run this after training/evaluation to refresh the web interface
#

set -e

echo "Updating frontend with latest classification data..."

# Generate latest classifications
echo "1. Generating classifications from storage data..."
python3 analyze_training_simple.py --output environment_classifications.json

# Copy to React app source
echo "2. Copying to React app source..."
cp environment_classifications.json front/my-react-app/src/

# Rebuild React app
echo "3. Building React app..."
cd front/my-react-app
npm run build

# Copy to build storage directory
echo "4. Updating build storage..."
cd ../..
cp environment_classifications.json front/my-react-app/build/storage/

echo ""
echo "✓ Frontend updated successfully!"
echo ""
echo "To view the updated frontend:"
echo "  cd front/my-react-app"
echo "  npm start"
echo ""
echo "Or serve the build folder:"
echo "  npx serve -s front/my-react-app/build"
