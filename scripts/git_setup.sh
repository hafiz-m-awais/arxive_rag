#!/bin/bash
# scripts/git_setup.sh — Initialize git repo with hooks and branch strategy.
# Run once: bash scripts/git_setup.sh

set -e

echo "=== Initializing Git repository ==="

# 1. Init repo if not already
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized"
else
    echo "Git repo already exists"
fi

# 2. Create .env from example if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# 3. Create gitkeep files so empty dirs are tracked
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch logs/.gitkeep
echo "Created .gitkeep files"

# 4. Install pre-commit hook (runs tests before every commit)
mkdir -p .git/hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "--- Pre-commit: running tests ---"
python -m pytest tests/ -q --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
echo "--- Tests passed ---"
EOF
chmod +x .git/hooks/pre-commit
echo "Pre-commit hook installed (runs pytest before every commit)"

# 5. Set up branch strategy
git checkout -b main 2>/dev/null || git checkout main
echo "On branch: main"

# 6. Initial commit
git add .
git commit -m "feat: initial production RAG system setup

- BGE-M3 hybrid embedding (dense + sparse)
- Qdrant local vector store
- HyDE query expansion
- BGE-reranker-v2-m3 cross-encoder reranking
- Qwen2.5:7B via Ollama for generation
- Flask REST API with SSE streaming
- RAGAS evaluation pipeline
- Full test suite (pytest)
" 2>/dev/null || echo "Nothing to commit or already committed"

echo ""
echo "=== Git setup complete ==="
echo ""
echo "Recommended branch workflow:"
echo "  main        — stable, production-ready"
echo "  develop     — integration branch"
echo "  feature/*   — new features (e.g. feature/add-caching)"
echo "  fix/*       — bug fixes"
echo ""
echo "Create develop branch:"
echo "  git checkout -b develop"
echo ""
echo "Push to remote:"
echo "  git remote add origin https://github.com/yourusername/production-rag.git"
echo "  git push -u origin main"
