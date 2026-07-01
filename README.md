# Exercise Data Log 🏋️

A "field survey" style analytics dashboard for an exercise dataset, built with
**Streamlit** and **Plotly**. It's a Python port of a React/Recharts dashboard,
covering:

- **Distribution charts** — exercises by category, equipment, target muscle,
  and secondary muscles engaged, plus a category-share donut chart and an
  equipment × category heatmap.
- **Model performance** — three model report cards (target-muscle classifier,
  secondary-muscle multi-label predictor, content-based recommender), a
  per-class F1 bar chart, and a classifier confusion matrix.
- **Exercise Explorer** — a searchable, filterable, paginated table of
  individual exercises (name, category, equipment, target, muscle group).

## Screenshot

Run the app locally (see below) to see the dashboard — it uses a warm,
paper-and-ink "training log" theme with `Oswald` headings and `JetBrains Mono`
data labels.

## Project structure

```
exercise-dashboard/
├── app.py                     # Main Streamlit app
├── requirements.txt
├── src/
│   ├── data_loader.py         # JSON → pandas loading helpers (cached)
│   └── charts.py               # Plotly chart builders (bar, donut, heatmaps)
└── data/
    ├── exercises.json          # Aggregate totals: category/equipment/target/secondary
    ├── exercises_rows.json     # Per-exercise rows used by the Explorer table
    └── model_results.json      # Model cards, per-class F1, confusion matrix, heatmap cells
```

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/exercise-data-log.git
cd exercise-data-log

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Swapping in your own data

The app reads three JSON files from `data/`:

| File | Purpose |
|---|---|
| `exercises.json` | Aggregate totals used by the distribution charts (`total`, `category`, `equipment`, `target`, `secondary_muscles`) |
| `exercises_rows.json` | A list of per-exercise records: `name`, `category`, `equipment`, `target`, `muscle_group` — powers the Explorer table |
| `model_results.json` | Model report-card copy/metrics, `per_class_f1`, the `confusion` matrix cells, and the `heatmap` cells |

Replace any of these with your own data in the same shape and the app will
pick it up automatically — no code changes required. `exercises_rows.json`
ships with a representative sample (~170 rows spanning every category and
most equipment types) rather than the full logged dataset; drop in the
complete row list to power a larger explorer table.

## Deploying

The app is a standard Streamlit app and deploys as-is to
[Streamlit Community Cloud](https://streamlit.io/cloud) — point it at
`app.py` on your fork/clone of this repo — or any platform that can run
`streamlit run app.py` (Render, Fly.io, a Docker container, etc.).

## Tech stack

- [Streamlit](https://streamlit.io/) — app framework / UI
- [Plotly](https://plotly.com/python/) — charts (bar, donut, heatmaps)
- [pandas](https://pandas.pydata.org/) — data wrangling

## License

MIT — see [LICENSE](LICENSE).
