# GitHub Upload

From inside the repository folder:

```bash
git init
git add .
git commit -m "Initial MAX-SR V2 composite pipeline"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Before pushing, run:

```bash
python scripts/validate_notebook.py
git status
```

Check that large datasets, `.npy` arrays, GeoTIFFs and `.pt` checkpoints are not staged accidentally.

If you want checkpoints published later, use a release asset or Git LFS rather than normal Git history.
