"""Exercise Coach Starter

This Streamlit app can load data from:
- a local folder or CSV file
- a public GitHub repository URL

It automatically looks for:
- tabular files such as CSV/TSV
- image datasets organized by class folders

Once loaded, it can train a lightweight baseline model and let you test
predictions inside the app.
"""

from __future__ import annotations

import io
import pickle
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
TABULAR_EXTENSIONS = {".csv", ".tsv"}
DEFAULT_GITHUB_REPO_URL = "https://github.com/bonsoul/exercises-dataset"
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


def running_under_streamlit() -> bool:
    """Detect whether the script is executing inside Streamlit runtime."""

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    return get_script_run_ctx(suppress_warning=True) is not None


def cache_data(*decorator_args, **decorator_kwargs):
    """Apply Streamlit caching only when the app is running under Streamlit."""

    if running_under_streamlit():
        return st.cache_data(*decorator_args, **decorator_kwargs)

    def decorator(func):
        return func

    return decorator


@dataclass(frozen=True)
class GitHubRepo:
    owner: str
    name: str
    branch_hint: str | None = None


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_target_column(columns: Iterable[str]) -> str | None:
    lookup = {str(col).lower(): col for col in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_tabular_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TABULAR_EXTENSIONS


def parse_github_repo_url(url: str) -> GitHubRepo:
    """Extract owner/repo and an optional branch hint from a GitHub URL."""

    parsed = urlparse(url.strip())
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise ValueError("Please enter a public GitHub repository URL.")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("That GitHub URL does not look like a repository URL.")

    owner = parts[0]
    name = parts[1].removesuffix(".git")
    branch_hint = None
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        branch_hint = parts[3]

    return GitHubRepo(owner=owner, name=name, branch_hint=branch_hint)


def download_and_extract_github_repo(repo_url: str, preferred_branch: str | None = None) -> tuple[Path, str]:
    """Download a public GitHub repository zip and extract it locally."""

    repo = parse_github_repo_url(repo_url)
    branch_candidates = []
    for branch in (repo.branch_hint, preferred_branch, "main", "master"):
        if branch and branch not in branch_candidates:
            branch_candidates.append(branch)

    last_error: Exception | None = None
    for branch in branch_candidates:
        zip_url = f"https://github.com/{repo.owner}/{repo.name}/archive/refs/heads/{branch}.zip"
        try:
            request = Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                zip_bytes = response.read()

            extract_dir = Path(tempfile.mkdtemp(prefix=f"{repo.name}-{branch}-"))
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                archive.extractall(extract_dir)

            children = [child for child in extract_dir.iterdir() if child.is_dir()]
            root = children[0] if len(children) == 1 else extract_dir
            return root, branch
        except (HTTPError, URLError, zipfile.BadZipFile, OSError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not download {repo.owner}/{repo.name} from GitHub. "
        f"Last error: {last_error}"
    )


@cache_data(show_spinner=False)
def load_csv_file(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def discover_tabular_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if is_tabular_file(root) else []
    return sorted([path for path in root.rglob("*") if is_tabular_file(path)])


def discover_image_dataset_roots(root: Path, min_images: int = 2) -> list[dict]:
    """Rank plausible image dataset roots by image count and depth."""

    if root.is_file():
        return []

    image_files = [path for path in root.rglob("*") if is_image_file(path)]
    stats: dict[Path, dict[str, object]] = {}

    for image_path in image_files:
        ancestors = [image_path.parent, *image_path.parents[1:]]
        for ancestor in ancestors:
            if ancestor == ancestor.parent:
                break
            try:
                rel = image_path.relative_to(ancestor)
            except ValueError:
                continue
            if len(rel.parts) < 2:
                continue

            info = stats.setdefault(ancestor, {"count": 0, "labels": set()})
            info["count"] = int(info["count"]) + 1
            labels = info["labels"]
            assert isinstance(labels, set)
            labels.add(rel.parts[0])

    candidates: list[dict] = []
    for candidate_root, info in stats.items():
        labels = info["labels"]
        assert isinstance(labels, set)
        count = int(info["count"])
        if count < min_images or len(labels) < 2:
            continue
        try:
            depth = len(candidate_root.relative_to(root).parts)
        except ValueError:
            continue
        candidates.append(
            {
                "path": candidate_root,
                "image_count": count,
                "label_count": len(labels),
                "depth": depth,
            }
        )

    candidates.sort(key=lambda item: (item["image_count"], item["depth"]), reverse=True)
    return candidates


def discover_image_samples(root: Path, dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str | None]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not is_image_file(path):
            continue
        try:
            rel_parts = path.relative_to(dataset_root).parts
        except ValueError:
            continue
        label = rel_parts[0] if len(rel_parts) > 1 else None
        rows.append({"path": str(path), "label": label})
    return pd.DataFrame(rows)


def image_to_vector(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    return (array / 255.0).reshape(-1)


def uploaded_image_to_vector(uploaded_file, image_size: int) -> np.ndarray:
    with Image.open(io.BytesIO(uploaded_file.getvalue())) as image:
        array = np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.float32)
    return (array / 255.0).reshape(-1)


def safe_train_test_split(X, y, test_size: float):
    y_series = pd.Series(y)
    stratify = None
    if y_series.nunique() > 1 and y_series.value_counts().min() >= 2:
        stratify = y

    try:
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=42)


def build_tabular_pipeline(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
) -> tuple[Pipeline, dict, list[str]]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

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
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=test_size)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": pd.DataFrame(
            classification_report(y_test, predictions, output_dict=True, zero_division=0)
        ).T,
        "confusion": pd.crosstab(
            pd.Series(y_test, name="actual"),
            pd.Series(predictions, name="predicted"),
            dropna=False,
        ),
    }
    feature_columns = X.columns.tolist()
    return pipeline, metrics, feature_columns


def build_image_model(samples: pd.DataFrame, image_size: int, test_size: float) -> tuple[Pipeline, dict]:
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
        raise ValueError("Not enough labeled, readable images to train a baseline model.")

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
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": pd.DataFrame(
            classification_report(y_test, predictions, output_dict=True, zero_division=0)
        ).T,
        "skipped": skipped,
    }
    pipeline = Pipeline(steps=[("model", model)])
    return pipeline, metrics


def align_tabular_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned.loc[:, feature_columns]


def render_inventory(root: Path) -> None:
    files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    if not files:
        st.warning("The loaded repo/folder does not contain any files.")
        return

    ext_counts = Counter(path.suffix.lower() or "[no ext]" for path in files)
    inventory_df = pd.DataFrame(
        sorted(ext_counts.items(), key=lambda item: item[1], reverse=True),
        columns=["extension", "count"],
    )
    st.markdown("#### File inventory")
    st.dataframe(inventory_df, use_container_width=True)

    preview = pd.DataFrame({"path": [str(path) for path in files[:50]]})
    st.markdown("#### First files")
    st.dataframe(preview, use_container_width=True)


def render_tabular_mode(df: pd.DataFrame, source_label: str, test_size: float) -> None:
    st.subheader("Tabular dataset")
    st.caption(f"Source: {source_label}")

    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Columns", f"{df.shape[1]:,}")
    cols[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    target_default = infer_target_column(df.columns)
    if target_default and target_default in df.columns:
        cols[3].metric("Classes", f"{df[target_default].nunique(dropna=True):,}")
    else:
        cols[3].metric("Classes", "n/a")

    target_index = 0
    if target_default in df.columns:
        target_index = list(df.columns).index(target_default)

    target_col = st.selectbox(
        "Choose the target column",
        options=list(df.columns),
        index=target_index,
    )

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("#### Preview")
        st.dataframe(df.head(25), use_container_width=True)
    with right:
        st.markdown("#### Data profile")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing.head(10).to_frame("missing"), use_container_width=True)
        else:
            st.info("No missing values detected.")

        st.write("Target distribution")
        st.dataframe(df[target_col].value_counts(dropna=False).head(15).to_frame("count"), use_container_width=True)

    if st.button("Train tabular baseline model", type="primary"):
        with st.spinner("Training tabular model..."):
            pipeline, metrics, feature_columns = build_tabular_pipeline(df, target_col, test_size=test_size)
            st.session_state["tabular_bundle"] = {
                "model": pipeline,
                "target_col": target_col,
                "feature_columns": feature_columns,
                "metrics": metrics,
            }
        st.success("Tabular model trained.")

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
                probabilities = bundle["model"].predict_proba(aligned)
                result["confidence"] = np.round(probabilities.max(axis=1), 3)

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


def render_image_mode(samples: pd.DataFrame, source_label: str, image_size: int, test_size: float) -> None:
    st.subheader("Image dataset")
    st.caption(f"Source: {source_label}")

    labeled = samples[samples["label"].notna()].copy()
    cols = st.columns(4)
    cols[0].metric("Images", f"{len(samples):,}")
    cols[1].metric("Labeled", f"{len(labeled):,}")
    cols[2].metric("Classes", f"{labeled['label'].nunique() if not labeled.empty else 0:,}")
    cols[3].metric("Unlabeled", f"{samples['label'].isna().sum():,}")

    if labeled.empty:
        st.warning("No labels were inferred. Put images inside class folders.")
    else:
        st.write("Class distribution")
        st.dataframe(labeled["label"].value_counts().to_frame("count"), use_container_width=True)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("#### Sample images")
        preview = samples.head(12)
        if preview.empty:
            st.info("No images found in the selected folder.")
        else:
            img_cols = st.columns(3)
            for idx, row in enumerate(preview.itertuples(index=False)):
                with img_cols[idx % 3]:
                    try:
                        st.image(Image.open(row.path), caption=row.label or "unlabeled")
                    except Exception:
                        st.write(f"Could not open: {row.path}")
    with right:
        st.markdown("#### Notes")
        st.write(
            "- The image model is a fast baseline using flattened pixels.\n"
            "- It is good for testing, not for production accuracy.\n"
            "- You can later replace it with a CNN."
        )

    trainable = labeled["label"].nunique() >= 2 and len(labeled) >= 4
    if st.button("Train image baseline model", type="primary", disabled=not trainable):
        with st.spinner("Training image model..."):
            pipeline, metrics = build_image_model(labeled[["path", "label"]], image_size=image_size, test_size=test_size)
            st.session_state["image_bundle"] = {
                "model": pipeline,
                "image_size": image_size,
                "metrics": metrics,
            }
        st.success("Image model trained.")

    bundle = st.session_state.get("image_bundle")
    if bundle:
        st.markdown("#### Results")
        st.metric("Accuracy", f"{bundle['metrics']['accuracy']:.3f}")
        st.caption(f"Skipped unreadable images: {bundle['metrics']['skipped']}")
        st.dataframe(bundle["metrics"]["report"], use_container_width=True)

        st.markdown("#### Predict on uploaded images")
        uploaded_images = st.file_uploader(
            "Upload one or more images",
            type=sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS),
            accept_multiple_files=True,
            key="image_predict_upload",
        )
        if uploaded_images:
            rows = []
            model = bundle["model"].named_steps["model"]
            for uploaded in uploaded_images:
                try:
                    vector = uploaded_image_to_vector(uploaded, bundle["image_size"])
                    prediction = model.predict([vector])[0]
                    row = {"file": uploaded.name, "prediction": prediction}
                    if hasattr(model, "predict_proba"):
                        row["confidence"] = float(np.max(model.predict_proba([vector])[0]))
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


def resolve_source() -> tuple[Path | None, str | None]:
    """Load either a local path or a GitHub repo and return a dataset root."""

    with st.sidebar:
        st.header("Data source")
        source_mode = st.radio("Choose source", ["GitHub repo", "Local path"], horizontal=False)

        if source_mode == "GitHub repo":
            repo_url = st.text_input("GitHub repo URL", value=DEFAULT_GITHUB_REPO_URL)
            branch = st.text_input("Preferred branch", value="main")

            if st.button("Load GitHub repo", type="primary"):
                with st.spinner("Downloading repository..."):
                    root, loaded_branch = download_and_extract_github_repo(repo_url, preferred_branch=branch)
                st.session_state["dataset_root"] = str(root)
                st.session_state["dataset_label"] = f"{repo_url} (branch: {loaded_branch})"
                st.session_state["source_mode"] = "GitHub repo"

        else:
            path_text = st.text_input(
                "Local folder or CSV path",
                value="data",
                help="Use a folder, a CSV file, or a GitHub-repo extraction folder.",
            )
            path = Path(path_text).expanduser()
            if path.exists():
                st.session_state["dataset_root"] = str(path)
                st.session_state["dataset_label"] = str(path)
                st.session_state["source_mode"] = "Local path"
            else:
                st.info("Enter a valid local path to continue.")

        if st.button("Clear loaded data"):
            st.session_state.pop("dataset_root", None)
            st.session_state.pop("dataset_label", None)
            st.session_state.pop("tabular_bundle", None)
            st.session_state.pop("image_bundle", None)
            st.rerun()

    root_text = st.session_state.get("dataset_root")
    label = st.session_state.get("dataset_label")
    if not root_text:
        return None, None
    return Path(root_text), label


def main() -> None:
    st.set_page_config(page_title="Exercise Coach Starter", layout="wide")
    st.title("Exercise Coach Starter")
    st.write(
        "Load a public GitHub repo or a local folder, then explore the data and train a quick baseline model."
    )

    dataset_root, source_label = resolve_source()
    if dataset_root is None:
        st.info(
            "Load the GitHub repo you attached, or point the app at a local dataset folder to continue."
        )
        st.stop()

    st.success(f"Loaded: {source_label}")

    tabular_files = discover_tabular_files(dataset_root)
    image_roots = discover_image_dataset_roots(dataset_root)
    render_inventory(dataset_root)

    if tabular_files and image_roots:
        data_mode = st.radio("Detected data types", ["Tabular", "Images"], horizontal=True)
    elif tabular_files:
        data_mode = "Tabular"
    elif image_roots:
        data_mode = "Images"
    else:
        st.warning(
            "No CSV/TSV or image folders were detected in that repo/folder. "
            "The file inventory above shows what is available."
        )
        st.stop()

    st.divider()

    if data_mode == "Tabular":
        chosen_file = tabular_files[0]
        if len(tabular_files) > 1:
            chosen_file = st.selectbox(
                "Choose a tabular file",
                options=tabular_files,
                format_func=lambda path: str(path.relative_to(dataset_root)),
            )
        df = load_csv_file(str(chosen_file))
        render_tabular_mode(df, source_label=str(chosen_file), test_size=0.2)
    else:
        selected_root = image_roots[0]["path"]
        if len(image_roots) > 1:
            selected_root = st.selectbox(
                "Choose the image dataset root",
                options=[item["path"] for item in image_roots],
                format_func=lambda path: str(path.relative_to(dataset_root)),
            )
        samples = discover_image_samples(dataset_root, selected_root)
        render_image_mode(
            samples,
            source_label=str(selected_root),
            image_size=64,
            test_size=0.2,
        )


if __name__ == "__main__":
    if running_under_streamlit():
        main()
    else:
        print("Run this app with: streamlit run streamlit_app.py")
