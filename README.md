# Exercise Coach Starter

Run the app:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Supported dataset layouts:

1. CSV/tabular data with a target column such as `label`, `class`, `target`, or `exercise`.
2. Images organized by class folders, such as `data/squat/*.jpg` and `data/pushup/*.jpg`.

The app includes:

- data profiling
- a baseline model
- batch prediction
- downloadable results
