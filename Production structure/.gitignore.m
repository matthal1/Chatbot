# Never commit API keys or passwords
.env
.env.local
config/secrets.py


# Git crashes with large files. These should be re-generated or downloaded, not versioned.
artifacts/
data/raw/
data/processed/
chroma_db_data/

# These change every time you run the code; you don't want them in history.
logs/
*.log
*.csv

# Python generates these automatically; they are useless to other people.
__pycache__/
*.pyc

# Your virtual environment folder
venv/
env/
xgb_env/

# Mac specific
.DS_Store