"""Model-specific prompt effects for the pooled MMLU PE-EmbPT replay."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import types
import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROMPTEVAL = REPO / "prompteval"
MATRIX_DIR = REPO / "data" / "MMLU_matrices"
YS_PATH = PROMPTEVAL / "data" / "Ys.pickle"
XS_PATH = PROMPTEVAL / "data" / "Xs.pickle"
FIRST_VERSION = REPO / "PE_EmbPT_data_2026-11-07" / "provenance" / "subject_sources.json"
VARIANTS = ("rasch389_random", "rasch389_balanced")
SEEDS = tuple(range(20))
N_MODELS, N_PROMPTS, N_EMBEDDING = 15, 100, 25
N_ARMS = N_MODELS * N_PROMPTS
X39_WIDTH = N_EMBEDDING + N_MODELS - 1
X389_WIDTH = N_MODELS * N_EMBEDDING + N_MODELS - 1
STATE: dict = {}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{contiguous.dtype.str}:{contiguous.shape}".encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_interaction_features(x39: np.ndarray) -> np.ndarray:
    """Place the 25 prompt coordinates in the corresponding model's block."""
    if x39.shape != (N_ARMS, X39_WIDTH) or not np.isfinite(x39).all():
        raise ValueError(f"Expected a finite {(N_ARMS, X39_WIDTH)} arm-feature design.")
    embeddings = x39[:, :N_EMBEDDING].reshape(N_MODELS, N_PROMPTS, N_EMBEDDING)
    if not np.all(embeddings == embeddings[0]):
        raise ValueError("Prompt features must agree across the 15 model blocks.")
    offsets = x39[:, N_EMBEDDING:]
    reference_coded = np.repeat(np.eye(N_MODELS)[:, 1:], N_PROMPTS, axis=0)
    if not np.array_equal(offsets, reference_coded):
        raise ValueError("The last 14 columns must be the original reference-coded model indicators.")

    model_index = np.repeat(np.arange(N_MODELS), N_PROMPTS)
    block_columns = model_index[:, None] * N_EMBEDDING + np.arange(N_EMBEDDING)
    x389 = np.zeros((N_ARMS, X389_WIDTH), dtype=np.float64)
    x389[np.arange(N_ARMS)[:, None], block_columns] = x39[:, :N_EMBEDDING]
    x389[:, N_MODELS * N_EMBEDDING:] = offsets
    return x389


def register_design(features: np.ndarray) -> float:
    """Validate native centered rank once and cache sparse rows for this subject."""
    pca = PCA().fit(features)
    min_ratio = float(pca.explained_variance_ratio_.min())
    if not min_ratio > 1e-6:
        raise AssertionError(f"The native PCA rank check failed ({min_ratio:g} <= 1e-6).")
    STATE["features"] = features
    STATE["csr_features"] = sparse.csr_matrix(features)
    return min_ratio


def check_registered_design(features: np.ndarray, tol: float = 1e-6) -> None:
    if STATE.get("features") is not features or tol != 1e-6:
        raise ValueError("Register the subject's unchanged Rasch arm design before fitting.")


def sparse_rasch_design(seen_items, y, x, z):
    """Use the same observed row-major design as native GenXY."""
    check_registered_design(x)
    n_questions = y.shape[1]
    if z.shape != (n_questions, n_questions) or np.count_nonzero(z) != n_questions or not np.all(z.diagonal() == 1):
        raise ValueError("The Rasch replay requires question identity features.")
    arms, questions = np.nonzero(seen_items)
    question_features = sparse.csr_matrix(
        (np.ones(len(arms)), questions, np.arange(len(arms) + 1)),
        shape=(len(arms), n_questions),
    )
    # Select rows after sparsifying the 1,500-arm design, never from a dense
    # observations-by-389 array with mostly inactive model-block columns.
    design = sparse.hstack((STATE["csr_features"][arms], question_features), format="csr")
    return design, y[arms, questions].astype(float)


def stratified_sample(seen_examples, max_seen, random_seed, active_arms=None, random_column=False):
    """Preserve native row/column ordering and both RandomState draws."""
    active = np.arange(seen_examples.shape[0]) if active_arms is None else np.asarray(active_arms, dtype=np.intp)
    rng = np.random.RandomState(random_seed)
    row_counts = seen_examples.sum(axis=1)
    column_counts = None if random_column else seen_examples.sum(axis=0)
    n_seen = int(row_counts.sum())
    while n_seen < max_seen:
        active_counts = row_counts[active]
        row = rng.choice(active[active_counts == active_counts.min()])
        available = np.flatnonzero(~seen_examples[row])
        if not available.size:
            break
        if column_counts is not None:
            counts = column_counts[available]
            available = available[counts == counts.min()]
        column = rng.choice(available)
        seen_examples[row, column] = True
        row_counts[row] += 1
        if column_counts is not None:
            column_counts[column] += 1
        n_seen += 1
    return seen_examples


class RaschScore:
    def fit(self, seen, matrix, features):
        model = STATE["methods"].ExtendedRaschModel()
        model.fit(seen, matrix, features, Z=None)
        self.thetas = model.get_Y_hat().mean(axis=1)


def phase_sampler(seen, max_seen, seed, active_arms, _random_column, *, random_column):
    return stratified_sample(seen, max_seen, seed, active_arms, random_column)


def replay_case(matrix, features, budget, seed, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown interaction variant: {variant}")
    bai = STATE["bai"]
    old_model, old_sampler = bai.LogReg, bai.StratSample
    bai.LogReg = RaschScore
    bai.StratSample = partial(phase_sampler, random_column=variant == VARIANTS[0])
    try:
        return bai.compute_regrets(budget, matrix, features, random_seed=seed, backend="sklearn", costs=STATE["prices"])
    finally:
        bai.LogReg, bai.StratSample = old_model, old_sampler


def initialize_worker() -> None:
    # Native methods only use utils for the centered rank check. Avoid importing
    # the unrelated encoder and Torch dependencies of prompteval/utils.py.
    sys.path.insert(0, str(PROMPTEVAL))
    utils = types.ModuleType("utils")
    utils.check_multicolinearity = check_registered_design
    sys.modules["utils"] = utils
    import bai_evaluation as bai
    import methods

    methods.GenXY = sparse_rasch_design
    methods.check_multicolinearity = check_registered_design
    prices_path = REPO / bai.DEFAULT_COST_FILES["MMLU"]
    inputs = (YS_PATH, XS_PATH, prices_path)
    source_files = (
        HERE / "interaction.py", HERE / "run_interaction.py",
        *(PROMPTEVAL / name for name in ("bai_evaluation.py", "methods.py", "methods_gpu.py", "utils.py")),
    )
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "tqdm", "threadpoolctl"):
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD")
        packages[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
        with YS_PATH.open("rb") as handle:
            ys = pickle.load(handle)
        with XS_PATH.open("rb") as handle:
            xs = pickle.load(handle)
    first_version = {}
    if FIRST_VERSION.is_file():
        first_version = json.loads(FIRST_VERSION.read_text(encoding="utf-8"))["subjects"]

    STATE.update(
        bai=bai, methods=methods, ys=ys, xs=xs,
        prices=bai.load_arm_costs(str(prices_path)), prices_path=prices_path,
        first_version=first_version, thread_limit=threadpool_limits(limits=1),
        provenance={
            "version": 1,
            "python_version": sys.version,
            "packages": packages,
            "source_sha256": {path.relative_to(REPO).as_posix(): sha256(path) for path in source_files},
            "input_sha256": {path.relative_to(REPO).as_posix(): sha256(path) for path in inputs},
            "fit": "Native L2 LIBLINEAR logistic regression at C=100, with no separate intercept and no feature rescaling.",
            "scoring": "Native PE row mean of observed labels and fitted probabilities for unseen questions.",
            "allocation": "Joint 1,500-arm successive halving with five phases and 10% of response cells.",
            "cost_policy": "Count observations for allocation and budget; track price-weighted spend from the same trajectories.",
            "arm_features": "15 model-specific blocks of 25 pretrained prompt coordinates, followed by the unchanged 14 model indicators.",
            "question_features": "One indicator per question, shared across all model-prompt arms.",
            "samplers": {
                VARIANTS[0]: "Uniform unseen question among the least-sampled active arm's available questions.",
                VARIANTS[1]: "Uniform among that arm's least-covered unseen questions, counting observations across all rows.",
            },
            "backend": "sklearn",
            "budget_fraction": bai.OBSERVATION_BUDGET_MAX_FRACTION,
            "phase_cap": bai.MAX_BAI_PHASES,
            "feature_names": VARIANTS,
        },
    )


def prepare_subject(subject: str):
    bai = STATE["bai"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MMLU/.*: Xs view .* right-padded")
        matrix, views = bai.combine_models_y_xs(STATE["ys"], STATE["xs"], "MMLU", subject)
    matrix_path = MATRIX_DIR / f"{subject}.npy"
    if not np.array_equal(matrix, np.load(matrix_path, allow_pickle=False)):
        raise ValueError(f"Stacked PromptEval response rows differ from {matrix_path}.")
    x39 = views[1]
    x389 = build_interaction_features(x39)
    rank = register_design(x389)
    x39_hash, x389_hash, matrix_hash = digest_array(x39), digest_array(x389), sha256(matrix_path)
    prior = STATE["first_version"].get(subject)
    if prior and (prior["shared_X39_sha256"], prior["matrix_sha256"]) != (x39_hash, matrix_hash):
        raise ValueError(f"{subject}: X39 or the response matrix differs from the first version.")

    provenance = STATE["provenance"] | {
        "subject": subject,
        "matrix_shape": list(matrix.shape),
        "matrix_sha256": matrix_hash,
        "X39_sha256": x39_hash,
        "X389_sha256": x389_hash,
        "arm_feature_width": X389_WIDTH,
        "question_feature_width": matrix.shape[1],
        "fitted_coefficients": X389_WIDTH + matrix.shape[1],
        "observation_budget": bai.budgets_from_y(matrix)[0],
        "rank_min_explained_variance_ratio": rank,
        "first_version_input_match": bool(prior),
    }
    return matrix, x389, provenance, digest_json(provenance)


@contextmanager
def one_thread_workers():
    variables = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    original = {name: os.environ.get(name) for name in variables}
    try:
        os.environ.update({name: "1" for name in variables})
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
