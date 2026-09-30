# SmartSpend AI

## Run
Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.

## Dataset location
Place `SmartSpendAI_cleaned_dataset.csv` beside `app.py` or inside `data/`. The app also accepts the filename `SmartSpendAI_cleaned_dataset-1.csv`.

This version keeps the app in one Python file to avoid cross-file import errors. Added transactions are session-only until you download the updated CSV.
