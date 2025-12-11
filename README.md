# Crypto Volatility — Demo

This repo contains a small ML pipeline: data cleaning, feature engineering, baseline models, and a Streamlit demo.

## How to run locally
1. Create and activate virtualenv (Python 3.11 recommended).
2. Install:
pip install -r requirements.txt

3. Train models:
python src/model.py

4. Run demo:
streamlit run src/app_streamlit.py


Notes:
- Large files (models/data) are not included by default. If you want to include them, remove the corresponding entries from `.gitignore` or host them externally (Google Drive / S3 / GitHub Releases).


Save the file.