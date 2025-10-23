#!/bin/bash
# This script opens your browser to create the Pull Request

echo "Opening GitHub to create Pull Request..."
echo ""
echo "Branch: claude/olmo2-transparency-pipeline-011CUN7o3TMUCpZvZk8FK2x6"
echo "Target: main"
echo ""

# Construct the URL
REPO="BarzinL/TranspOLMo2-1B"
BRANCH="claude/olmo2-transparency-pipeline-011CUN7o3TMUCpZvZk8FK2x6"
URL="https://github.com/${REPO}/compare/main...${BRANCH}?expand=1"

echo "Opening: $URL"
echo ""

# Try to open in browser
if command -v xdg-open &> /dev/null; then
    xdg-open "$URL"
elif command -v open &> /dev/null; then
    open "$URL"
else
    echo "Please manually open this URL in your browser:"
    echo "$URL"
fi

echo ""
echo "The PR description is in: PR_DESCRIPTION.md"
echo "Copy its contents into the PR description field."
echo ""
echo "Title: TranspOLMo: First-Principles Neural Network Interpretability Framework"
