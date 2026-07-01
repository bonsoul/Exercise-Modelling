"""Exercise Coach Starter

This Streamlit app is a lightweight starter for an exercise dataset product.

It supports two common dataset layouts:

1. Tabular data in CSV files with a label column such as `label`, `class`,
   `target`, or `exercise`.
2. Image datasets organized by class folder, for example:

   data/
     squat/
       img001.jpg
       img002.jpg
     pushup/
       img101.jpg

The app gives you:
- dataset exploration
- a simple baseline model
- batch prediction for new CSV rows or images

It is intentionally dependency-light so you can turn it into a real MVP fast.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
IMAGE_UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)
TARGET_CANDIDATES = (
    "label",
    "class",
    "target",
    "exercise",
    "activity",
    "movement",
    "category",
    "outcome",
)


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both older and newer scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_target_column(columns: Sequence[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def find_first_csv(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
        return path

    candidates = sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}]
    )
    return candidates[0] if candidates else None


@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


@st.cache_data(show_spinner=False)
def discover_image_samples(root_path: str) -> pd.DataFrame:
    root = Path(root_path)
    rows: list[dict[str, str | None]] = []

    for path in sorted(root.rglob("*")):
        if not is_image_file(path):
            continue

        rel_parts = path.relative_to(root).parts
        label = rel_parts[0] if len(rel_parts) > 1 else None
        rows.append({"path": str(path), "label": label})

    return pd.DataFrame(rows)


def image_to_vector(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    return (arr / 255.0).reshape(-1)


def uploaded_image_to_vector(uploaded_file, image_size: int) -> np.ndarray:
    data = uploaded_file.getvalue()
    with Image.open(io.BytesIO(data)) as img:
        arr = np.asarray(img.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    return (arr / 255.0).reshape(-1)


def safe_train_test_split(X, y, test_size: float):
    y_series = pd.Series(y)
    stratify = None
    if y_series.nunique() > 1 and y_series.value_counts().min() >= 2:
        stratify = y

    try:
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=42)


def build_tabular_pipeline(df: pd.DataFrame, target_col: str) -> tuple[Pipeline, dict, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.2)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": pd.DataFrame(
            classification_report(y_test, preds, output_dict=True, zero_division=0)
        ).T,
        "confusion": pd.crosstab(
            pd.Series(y_test, name="actual"),
            pd.Series(preds, name="predicted"),
            dropna=False,
        ),
    }
    feature_columns = X.columns.tolist()
    return pipeline, metrics, X_test, y_test, preds, feature_columns


def align_tabular_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    aligned = df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned.loc[:, list(feature_columns)]


def sample_image_frame(frame: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    if frame.empty or "label" not in frame.columns:
        return frame

    labeled = frame[frame["label"].notna()].copy()
    if labeled.empty or max_per_class <= 0:
        return labeled

    return (
        labeled.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), max_per_class), random_state=42))
        .reset_index(drop=True)
    )


def build_image_model(samples: pd.DataFrame, image_size: int, test_size: float) -> tuple[Pipeline, dict, np.ndarray, np.ndarray, np.ndarray]:
    vectors = []
    labels = []
    skipped = 0

    for row in samples.itertuples(index=False):
        if pd.isna(row.label):
            continue
        try:
            vectors.append(image_to_vector(row.path, image_size))
            labels.append(row.label)
        except Exception:
            skipped += 1

    if len(vectors) < 2:
        raise ValueError("Not enough readable labeled images to train a model.")

    X = np.asarray(vectors)
    y = np.asarray(labels)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=test_size)

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2500,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": pd.DataFrame(
            classification_report(y_test, preds, output_dict=True, zero_division=0)
        ).T,
        "skipped": skipped,
    }
    pipeline = Pipeline(steps=[("model", model)])
    return pipeline, metrics, X_test, y_test, preds


def render_overview_cards(df: pd.DataFrame, target_col: str | None = None) -> None:
    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Columns", f"{df.shape[1]:,}")
    cols[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    if target_col and target_col in df.columns:
        cols[3].metric("Classes", f"{df[target_col].nunique(dropna=True):,}")
    else:
        cols[3].metric("Classes", "n/a")


def render_tabular_mode(df: pd.DataFrame, source_label: str, test_size: float) -> None:
    st.subheader("Tabular dataset")
    st.caption(f"Source: {source_label}")
    render_overview_cards(df)

    target_default = infer_target_column(df.columns)
    target_col = st.selectbox(
        "Choose the target column",
        options=list(df.columns),
        index=list(df.columns).index(target_default) if target_default in df.columns else 0,
    )

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("#### Preview")
        st.dataframe(df.head(25), use_container_width=True)
    with right:
        st.markdown("#### Data profile")
        st.write("Top missing columns")
        missing = df.isna().sum().sort_values(ascending=False)
        st.dataframe(missing[missing > 0].head(10).to_frame("missing"), use_container_width=True)
        st.write("Target distribution")
        st.dataframe(df[target_col].value_counts(dropna=False).head(15).to_frame("count"), use_container_width=True)

    if st.button("Train tabular baseline model", type="primary"):
        with st.spinner("Training model..."):
            pipeline, metrics, X_test, y_test, preds, feature_columns = build_tabular_pipeline(df, target_col)
            st.session_state["tabular_bundle"] = {
                "model": pipeline,
                "target_col": target_col,
                "feature_columns": feature_columns,
                "metrics": metrics,
            }
        st.success("Model trained and stored in session state.")

    bundle = st.session_state.get("tabular_bundle")
    if bundle:
        st.markdown("#### Results")
        st.metric("Accuracy", f"{bundle['metrics']['accuracy']:.3f}")
        st.dataframe(bundle["metrics"]["report"], use_container_width=True)
        st.dataframe(bundle["metrics"]["confusion"], use_container_width=True)

        st.markdown("#### Batch prediction")
        uploaded = st.file_uploader(
            "Upload a CSV with the same feature columns",
            type=["csv"],
            key="tabular_predict_upload",
        )
        if uploaded is not None:
            new_df = pd.read_csv(uploaded, sep=None, engine="python")
            if bundle["target_col"] in new_df.columns:
                new_df = new_df.drop(columns=[bundle["target_col"]])
            aligned = align_tabular_frame(new_df, bundle["feature_columns"])
            predictions = bundle["model"].predict(aligned)
            result = new_df.copy()
            result["prediction"] = predictions

            if hasattr(bundle["model"], "predict_proba"):
                probs = bundle["model"].predict_proba(aligned)
                confidence = probs.max(axis=1)
                result["confidence"] = np.round(confidence, 3)

            st.dataframe(result.head(50), use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="exercise_predictions.csv",
                mime="text/csv",
            )

        st.download_button(
            "Download trained model",
            data=pickle.dumps(bundle),
            file_name="tabular_exercise_model.pkl",
            mime="application/octet-stream",
        )


def render_image_mode(samples: pd.DataFrame, source_label: str, image_size: int, test_size: float, max_per_class: int) -> None:
    st.subheader("Image dataset")
    st.caption(f"Source: {source_label}")

    sampled = sample_image_frame(samples, max_per_class=max_per_class)
    render_overview_cards(sampled, target_col="label")

    labeled = sampled[sampled["label"].notna()].copy()
    if labeled.empty:
        st.warning("No class labels were inferred. Organize images into class folders to train a model.")
    else:
        st.write("Class distribution")
        st.dataframe(labeled["label"].value_counts().to_frame("count"), use_container_width=True)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("#### Sample images")
        preview = sampled.head(12)
        if preview.empty:
            st.info("No images found in this folder.")
        else:
            cols = st.columns(3)
            for idx, row in enumerate(preview.itertuples(index=False)):
                with cols[idx % 3]:
                    try:
                        st.image(Image.open(row.path), caption=f"{row.label or 'unlabeled'}")
                    except Exception:
                        st.write(f"Could not open: {row.path}")
    with right:
        st.markdown("#### Notes")
        st.write(
            "- Put images in class folders for training.\n"
            "- The baseline model is a fast, simple starter.\n"
            "- You can replace it with a CNN later."
        )

    trainable = labeled["label"].nunique() >= 2 and len(labeled) >= 4
    if st.button("Train image baseline model", type="primary", disabled=not trainable):
        with st.spinner("Training model..."):
            pipeline, metrics, X_test, y_test, preds = build_image_model(
                labeled[["path", "label"]],
                image_size=image_size,
                test_size=test_size,
            )
            st.session_state["image_bundle"] = {
                "model": pipeline,
                "image_size": image_size,
                "metrics": metrics,
            }
        st.success("Model trained and stored in session state.")

    bundle = st.session_state.get("image_bundle")
    if bundle:
        st.markdown("#### Results")
        st.metric("Accuracy", f"{bundle['metrics']['accuracy']:.3f}")
        if "skipped" in bundle["metrics"]:
            st.caption(f"Skipped unreadable images: {bundle['metrics']['skipped']}")
        st.dataframe(bundle["metrics"]["report"], use_container_width=True)

        st.markdown("#### Predict on uploaded images")
        uploaded_images = st.file_uploader(
            "Upload one or more images",
            type=IMAGE_UPLOAD_TYPES,
            accept_multiple_files=True,
            key="image_predict_upload",
        )
        if uploaded_images:
            rows = []
            for uploaded in uploaded_images:
                try:
                    vector = uploaded_image_to_vector(uploaded, bundle["image_size"])
                    model = bundle["model"].named_steps["model"]
                    pred = model.predict([vector])[0]
                    row = {"file": uploaded.name, "prediction": pred}
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba([vector])[0]
                        row["confidence"] = float(np.max(probs))
                    rows.append(row)
                except Exception as exc:
                    rows.append({"file": uploaded.name, "prediction": f"error: {exc}"})

            pred_df = pd.DataFrame(rows)
            st.dataframe(pred_df, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="image_predictions.csv",
                mime="text/csv",
            )

        st.download_button(
            "Download trained model",
            data=pickle.dumps(bundle),
            file_name="image_exercise_model.pkl",
            mime="application/octet-stream",
        )


def main() -> None:
    st.set_page_config(page_title="Exercise Coach Starter", page_icon="🏋️", layout="wide")
    st.title("Exercise Coach Starter")
    st.write(
        "A fast MVP for turning an exercise dataset into a simple coach product: "
        "explore the data, train a baseline model, and test predictions."
    )

    with st.sidebar:
        st.header("Dataset source")
        source_text = st.text_input(
            "Path to your dataset folder or CSV",
            value="data",
            help="Example: data, data/exercises.csv, or a folder of class-organized images.",
        )
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        image_size = st.slider("Image resize", min_value=32, max_value=128, value=64, step=16)
        max_per_class = st.number_input(
            "Max images per class",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
            help="Limits how many images per label are used for preview/training.",
        )

    path = Path(source_text).expanduser()
    if not path.exists():
        st.info(
            "Point the sidebar at a dataset folder or CSV. "
            "For image mode, use folders like `data/squat/*.jpg`."
        )
        st.stop()

    csv_path = find_first_csv(path)
    if csv_path is not None:
        df = load_csv(str(csv_path))
        render_tabular_mode(df, source_label=str(csv_path), test_size=test_size)
        return

    image_samples = discover_image_samples(str(path))
    if not image_samples.empty:
        render_image_mode(
            image_samples,
            source_label=str(path),
            image_size=image_size,
            test_size=test_size,
            max_per_class=int(max_per_class),
        )
        return

    st.warning(
        "I could not detect a CSV file or any images. "
        "Drop a CSV into the folder, or organize images into class subfolders."
    )


if __name__ == "__main__":
    main()
