import os
import random
import warnings
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def set_random_seed(seed: int = 42) -> None:
    """Fix random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 1. DATA LOADING, SYNTHETIC FALLBACK & USER-PRESERVING SAMPLING
# =====================================================================

def generate_synthetic_kion_data(
    num_users: int = 2000,
    num_items: int = 800,
    num_interactions: int = 120000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate realistic KION-schema dataset if raw CSV files are absent."""
    rng = np.random.default_rng(seed)

    # Users table
    user_ids = np.arange(1, num_users + 1)
    age_buckets = rng.choice(["18_24", "25_34", "35_44", "45_54", "55+"], size=num_users)
    income_buckets = rng.choice(["low", "medium", "high"], size=num_users, p=[0.3, 0.5, 0.2])
    sex = rng.choice(["M", "F"], size=num_users)
    users_df = pd.DataFrame({
        "user_id": user_ids,
        "age": age_buckets,
        "income": income_buckets,
        "sex": sex,
    })

    # Items table
    item_ids = np.arange(1, num_items + 1)
    genres = rng.choice(
        ["action", "drama", "comedy", "scifi", "thriller", "documentary"],
        size=num_items,
    )
    content_type = rng.choice(["movie", "series"], size=num_items, p=[0.65, 0.35])
    release_year = rng.choice([2015, 2018, 2020, 2022, 2024], size=num_items)
    items_df = pd.DataFrame({
        "item_id": item_ids,
        "genre": genres,
        "content_type": content_type,
        "release_year": release_year,
    })

    # Latent preference affinity to make sequential and indirect signals meaningful
    user_pref_genre = {u: rng.choice(["action", "drama", "comedy", "scifi", "thriller", "documentary"]) for u in user_ids}
    item_genre_map = dict(zip(items_df["item_id"], items_df["genre"]))

    # Zipf-like item popularity
    item_weights = 1.0 / (np.arange(1, num_items + 1) ** 0.75)
    item_weights /= item_weights.sum()

    sampled_users = rng.choice(user_ids, size=num_interactions)
    sampled_items = rng.choice(item_ids, size=num_interactions, p=item_weights)

    # Timestamps over 90 days (in seconds)
    base_ts = 1700000000
    timestamps = base_ts + rng.integers(0, 90 * 86400, size=num_interactions)

    # Watched percentage and total duration
    genre_match = np.array([
        1.0 if user_pref_genre[u] == item_genre_map[i] else 0.0
        for u, i in zip(sampled_users, sampled_items)
    ])
    watched_pct = np.clip(
        rng.normal(loc=35.0 + 35.0 * genre_match, scale=25.0, size=num_interactions),
        0.0,
        100.0,
    ).round(1)
    total_dur = rng.integers(300, 7200, size=num_interactions) * (watched_pct / 100.0 + 0.1)

    interactions_df = pd.DataFrame({
        "user_id": sampled_users,
        "item_id": sampled_items,
        "timestamp": timestamps,
        "total_dur": total_dur.round(0),
        "watched_pct": watched_pct,
    }).sort_values("timestamp").reset_index(drop=True)

    # Deduplicate exact user-item-timestamp collisions
    interactions_df = interactions_df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    return interactions_df, users_df, items_df


def load_or_create_dataset(
    data_dir: str = "data",
    sample_users_count: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load KION CSV files if present, preserving full user histories upon sampling."""
    inter_path = os.path.join(data_dir, "interactions.csv")
    users_path = os.path.join(data_dir, "users.csv")
    items_path = os.path.join(data_dir, "items.csv")

    if os.path.exists(inter_path) and os.path.exists(users_path) and os.path.exists(items_path):
        print("Loading KION dataset from disk...")
        interactions = pd.read_csv(inter_path)
        users = pd.read_csv(users_path)
        items = pd.read_csv(items_path)

        if "last_watch_dt" in interactions.columns and "timestamp" not in interactions.columns:
            interactions["timestamp"] = pd.to_datetime(interactions["last_watch_dt"]).astype("int64") // 10**9

        if "genres" in items.columns and "genre" not in items.columns:
            items["genre"] = items["genres"].fillna("unknown").apply(lambda x: str(x).split(",")[0].strip())

        unique_users = interactions["user_id"].unique()
        if len(unique_users) > sample_users_count:
            rng = np.random.default_rng(seed)
            selected_users = set(rng.choice(unique_users, size=sample_users_count, replace=False))
            # Crucial rule: keep the ENTIRE interaction history for sampled users
            interactions = interactions[interactions["user_id"].isin(selected_users)].copy()
            users = users[users["user_id"].isin(selected_users)].copy()
    else:
        print("KION files not found in ./data. Generating realistic KION-schema dataset...")
        interactions, users, items = generate_synthetic_kion_data(
            num_users=sample_users_count, seed=seed
        )

    return interactions.sort_values("timestamp").reset_index(drop=True), users, items


# =====================================================================
# 2. TEMPORAL SPLITTING & CANDIDATE GENERATION (NO DATA LEAKAGE)
# =====================================================================

def temporal_split_and_candidates(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Create strict temporal splits:
    - [0%, 70%)   : Base history for train
    - [70%, 80%)  : Train target window
    - [80%, 90%)  : Validation target window (uses [0%, 80%) as history)
    - [90%, 100%) : Test target window (uses [0%, 90%) as history)

    Target binary label = 1 if user interacted with item and watched_pct >= 25.0 (or positive click/watch),
    0 for sampled negative candidates or bounced impressions.
    """
    rng = np.random.default_rng(seed)
    q70 = int(interactions["timestamp"].quantile(0.70))
    q80 = int(interactions["timestamp"].quantile(0.80))
    q90 = int(interactions["timestamp"].quantile(0.90))

    all_items = items["item_id"].unique()
    # Calculate item popularity strictly from initial history [0%, 70%) to avoid future leak in negative sampling
    hist_base = interactions[interactions["timestamp"] < q70]
    item_counts = hist_base["item_id"].value_counts()
    pop_items = item_counts.index.to_numpy()
    pop_probs = (item_counts.values ** 0.75) / (item_counts.values ** 0.75).sum()

    def build_labeled_candidates(history_df: pd.DataFrame, target_df: pd.DataFrame, split_ts: int) -> pd.DataFrame:
        active_users = set(history_df["user_id"].unique())
        target_df = target_df[target_df["user_id"].isin(active_users)].copy()

        # Positive interactions in target window (meaningful consumption: watched_pct >= 25%)
        pos_df = target_df[target_df["watched_pct"] >= 25.0][["user_id", "item_id"]].drop_duplicates()
        pos_df["label"] = 1

        # Negative candidates: combine low-watch bounces (<25%) and popularity-sampled negatives
        user_seen = history_df.groupby("user_id")["item_id"].apply(set).to_dict()
        user_target_pos = pos_df.groupby("user_id")["item_id"].apply(set).to_dict()

        neg_rows = []
        for user_id, pos_items in user_target_pos.items():
            n_neg = len(pos_items) * negatives_per_positive
            sampled = rng.choice(pop_items, size=n_neg * 2, replace=True, p=pop_probs)
            forbidden = user_seen.get(user_id, set()) | pos_items
            valid_neg = [it for it in sampled if it not in forbidden][:n_neg]
            for it in valid_neg:
                neg_rows.append({"user_id": user_id, "item_id": it, "label": 0})

        neg_df = pd.DataFrame(neg_rows)
        candidates = pd.concat([pos_df, neg_df], ignore_index=True)
        candidates["split_timestamp"] = split_ts
        return candidates.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    splits = {
        "train": (
            interactions[interactions["timestamp"] < q70].copy(),
            build_labeled_candidates(
                interactions[interactions["timestamp"] < q70],
                interactions[(interactions["timestamp"] >= q70) & (interactions["timestamp"] < q80)],
                q70,
            ),
            q70,
        ),
        "val": (
            interactions[interactions["timestamp"] < q80].copy(),
            build_labeled_candidates(
                interactions[interactions["timestamp"] < q80],
                interactions[(interactions["timestamp"] >= q80) & (interactions["timestamp"] < q90)],
                q80,
            ),
            q80,
        ),
        "test": (
            interactions[interactions["timestamp"] < q90].copy(),
            build_labeled_candidates(
                interactions[interactions["timestamp"] < q90],
                interactions[interactions["timestamp"] >= q90],
                q90,
            ),
            q90,
        ),
    }
    return splits


# =====================================================================
# 3. BASELINE: MANUAL FEATURE ENGINEERING
# =====================================================================

class ManualFeatureBuilder:
    """Computes user, item, and direct/indirect user-item features strictly from past history."""

    def __init__(self, users: pd.DataFrame, items: pd.DataFrame):
        self.users = users.copy()
        self.items = items.copy()
        self.cat_maps: Dict[str, Dict] = {}
        self._fit_categorical_encoders()

    def _fit_categorical_encoders(self) -> None:
        for col in ["age", "income", "sex"]:
            if col in self.users.columns:
                uniq = self.users[col].fillna("unk").unique()
                self.cat_maps[f"user_{col}"] = {v: i for i, v in enumerate(uniq)}
                self.users[f"user_{col}_code"] = self.users[col].fillna("unk").map(self.cat_maps[f"user_{col}"])

        for col in ["genre", "content_type"]:
            if col in self.items.columns:
                uniq = self.items[col].fillna("unk").unique()
                self.cat_maps[f"item_{col}"] = {v: i for i, v in enumerate(uniq)}
                self.items[f"item_{col}_code"] = self.items[col].fillna("unk").map(self.cat_maps[f"item_{col}"])

    def build_features(self, history: pd.DataFrame, candidates: pd.DataFrame, split_ts: int) -> pd.DataFrame:
        df = candidates.copy()
        hist = history[history["timestamp"] < split_ts].copy()
        hist = hist.merge(self.items[["item_id", "genre", "content_type"]], on="item_id", how="left")

        # 1. Static categorical attributes
        df = df.merge(
            self.users[["user_id", "user_age_code", "user_income_code", "user_sex_code"]],
            on="user_id",
            how="left",
        )
        df = df.merge(
            self.items[["item_id", "genre", "content_type", "release_year", "item_genre_code", "item_content_type_code"]],
            on="item_id",
            how="left",
        )

        # 2. Time-window aggregations (7 days, 30 days, full history)
        windows = {
            "7d": split_ts - 7 * 86400,
            "30d": split_ts - 30 * 86400,
            "all": 0,
        }

        for w_name, min_ts in windows.items():
            w_hist = hist[hist["timestamp"] >= min_ts]

            # User aggregations
            u_agg = w_hist.groupby("user_id").agg(
                **{
                    f"u_cnt_{w_name}": ("item_id", "count"),
                    f"u_watch_mean_{w_name}": ("watched_pct", "mean"),
                    f"u_dur_sum_{w_name}": ("total_dur", "sum"),
                }
            )
            df = df.merge(u_agg, on="user_id", how="left")

            # Item aggregations
            i_agg = w_hist.groupby("item_id").agg(
                **{
                    f"i_cnt_{w_name}": ("user_id", "count"),
                    f"i_watch_mean_{w_name}": ("watched_pct", "mean"),
                    f"i_high_watch_rate_{w_name}": ("watched_pct", lambda s: (s >= 50.0).mean()),
                }
            )
            df = df.merge(i_agg, on="item_id", how="left")

        # 3. User-Item direct and indirect interaction features
        # Direct repeats or prior views
        ui_direct = hist.groupby(["user_id", "item_id"]).agg(
            ui_direct_cnt=("timestamp", "count"),
            ui_direct_max_watch=("watched_pct", "max"),
        )
        df = df.merge(ui_direct, on=["user_id", "item_id"], how="left")

        # Indirect affinity: User <-> Genre and User <-> Content Type
        ug_agg = hist.groupby(["user_id", "genre"]).agg(
            ug_cnt=("item_id", "count"),
            ug_watch_mean=("watched_pct", "mean"),
            ug_dur_sum=("total_dur", "sum"),
        )
        df = df.merge(ug_agg, on=["user_id", "genre"], how="left")

        uc_agg = hist.groupby(["user_id", "content_type"]).agg(
            uc_cnt=("item_id", "count"),
            uc_watch_mean=("watched_pct", "mean"),
        )
        df = df.merge(uc_agg, on=["user_id", "content_type"], how="left")

        # Recency features
        u_last = hist.groupby("user_id")["timestamp"].max().rename("u_last_ts")
        i_last = hist.groupby("item_id")["timestamp"].max().rename("i_last_ts")
        df = df.merge(u_last, on="user_id", how="left")
        df = df.merge(i_last, on="item_id", how="left")
        df["u_recency_sec"] = split_ts - df["u_last_ts"].fillna(split_ts - 90 * 86400)
        df["i_recency_sec"] = split_ts - df["i_last_ts"].fillna(split_ts - 90 * 86400)

        # Manual conversion / affinity ratios
        df["ug_share_all"] = df["ug_cnt"].fillna(0) / (df["u_cnt_all"].fillna(0) + 1.0)
        df["i_trend_ratio_7d_30d"] = df["i_cnt_7d"].fillna(0) / (df["i_cnt_30d"].fillna(0) + 1.0)

        drop_helper_cols = ["genre", "content_type", "u_last_ts", "i_last_ts"]
        df = df.drop(columns=[c for c in drop_helper_cols if c in df.columns])
        return df.fillna(0.0)


# =====================================================================
# 4. RESEARCH DIRECTION 1: AUTOMATED SQL-LIKE FEATURE GENERATION & SELECTION
# =====================================================================

class AutomatedFeatureEngine:
    """
    Automatically synthesizes relational group-by aggregations, temporal decay metrics,
    and non-linear interaction ratios (SQL-style DSL templates), then selects optimal
    subset on validation set via collinearity pruning and LightGBM gain importance.
    """

    def __init__(self, items: pd.DataFrame, max_selected_features: int = 20):
        self.items = items[["item_id", "genre", "content_type", "release_year"]].copy()
        self.max_selected_features = max_selected_features
        self.selected_auto_cols: List[str] = []

    def synthesize_candidates(self, base_df: pd.DataFrame, history: pd.DataFrame, split_ts: int) -> pd.DataFrame:
        df = base_df.copy()
        hist = history[history["timestamp"] < split_ts].merge(self.items, on="item_id", how="left")

        # Template Group 1: Exponential time-decayed event weights (half-life = 3d, 14d)
        age_days = (split_ts - hist["timestamp"]) / 86400.0
        hist["decay_3d"] = np.exp(-0.231 * age_days)
        hist["decay_14d"] = np.exp(-0.0495 * age_days)
        hist["weighted_watch_3d"] = hist["decay_3d"] * hist["watched_pct"]
        hist["weighted_watch_14d"] = hist["decay_14d"] * hist["watched_pct"]

        # Template Group 2: Multi-level SQL-style GROUP BY over entities and attributes
        auto_group_specs = [
            (["user_id"], "auto_u"),
            (["item_id"], "auto_i"),
            (["user_id", "genre"], "auto_ug"),
            (["user_id", "content_type"], "auto_uc"),
            (["user_id", "release_year"], "auto_uy"),
        ]

        df = df.merge(self.items[["item_id", "genre", "content_type"]], on="item_id", how="left")

        for keys, prefix in auto_group_specs:
            agg_df = hist.groupby(keys).agg(
                **{
                    f"{prefix}_decay3d_sum": ("decay_3d", "sum"),
                    f"{prefix}_decay14d_sum": ("decay_14d", "sum"),
                    f"{prefix}_wwatch3d_mean": ("weighted_watch_3d", "mean"),
                    f"{prefix}_wwatch14d_mean": ("weighted_watch_14d", "mean"),
                    f"{prefix}_watch_std": ("watched_pct", "std"),
                    f"{prefix}_watch_max": ("watched_pct", "max"),
                    f"{prefix}_dur_mean": ("total_dur", "mean"),
                }
            )
            df = df.merge(agg_df, on=keys, how="left")

        # Template Group 3: Automated Relational Ratios & Cross-Entity Differences
        df["auto_ug_decay3d_share"] = df["auto_ug_decay3d_sum"].fillna(0) / (df["auto_u_decay3d_sum"].fillna(0) + 1e-4)
        df["auto_ug_decay14d_share"] = df["auto_ug_decay14d_sum"].fillna(0) / (df["auto_u_decay14d_sum"].fillna(0) + 1e-4)
        df["auto_uc_decay3d_share"] = df["auto_uc_decay3d_sum"].fillna(0) / (df["auto_u_decay3d_sum"].fillna(0) + 1e-4)
        df["auto_uy_decay14d_share"] = df["auto_uy_decay14d_sum"].fillna(0) / (df["auto_u_decay14d_sum"].fillna(0) + 1e-4)
        df["auto_user_vs_item_watch_diff"] = df["u_watch_mean_all"] - df["i_watch_mean_all"]
        df["auto_ug_vs_user_watch_ratio"] = (df["ug_watch_mean"].fillna(0) + 1.0) / (df["u_watch_mean_all"].fillna(0) + 1.0)
        df["auto_item_momentum_decay"] = df["auto_i_decay3d_sum"].fillna(0) / (df["auto_i_decay14d_sum"].fillna(0) + 1e-4)

        df = df.drop(columns=["genre", "content_type"]).fillna(0.0)
        return df

    def fit_select(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        manual_cols: List[str],
    ) -> List[str]:
        """Filter out highly collinear features and select top K via validation gain importance."""
        candidate_auto_cols = [c for c in train_df.columns if c.startswith("auto_")]

        # Step 1: Variance and correlation filter
        non_const = [c for c in candidate_auto_cols if train_df[c].std() > 1e-6]
        corr_matrix = train_df[non_const].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        filtered_auto = [col for col in upper.columns if not any(upper[col] > 0.95)]

        # Step 2: LightGBM Gain Importance ranking
        all_Candidate_features = manual_cols + filtered_auto
        selector_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        selector_model.fit(
            train_df[all_Candidate_features],
            train_df["label"],
            eval_set=[(val_df[all_Candidate_features], val_df["label"])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        importances = pd.Series(selector_model.booster_.feature_importance(importance_type="gain"), index=all_Candidate_features)
        auto_importances = importances[filtered_auto].sort_values(ascending=False)
        self.selected_auto_cols = auto_importances.head(self.max_selected_features).index.tolist()
        return self.selected_auto_cols


# =====================================================================
# 5. RESEARCH DIRECTION 3: TARGET-AWARE SEQUENCE MODEL (DIN / TRANSFORMER)
# =====================================================================

class UserHistorySequenceDataset(Dataset):
    """Packs user's recent K interactions (items, genres, watch_pct, time delta) and target item."""

    def __init__(
        self,
        history: pd.DataFrame,
        candidates: pd.DataFrame,
        items: pd.DataFrame,
        item_to_idx: Dict[int, int],
        genre_to_idx: Dict[str, int],
        split_ts: int,
        max_seq_len: int = 30,
    ):
        self.candidates = candidates.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.item_to_idx = item_to_idx
        self.genre_to_idx = genre_to_idx

        item_genre_map = dict(zip(items["item_id"], items["genre"].map(genre_to_idx)))
        self.item_genre_idx = {item_to_idx[k]: v for k, v in item_genre_map.items() if k in item_to_idx}

        # Filter history strictly prior to split_ts and truncate to last max_seq_len events per user
        hist = history[history["timestamp"] < split_ts].sort_values("timestamp")
        hist["item_idx"] = hist["item_id"].map(lambda x: item_to_idx.get(x, 0))
        hist["genre_idx"] = hist["item_idx"].map(lambda x: self.item_genre_idx.get(x, 0))
        hist["norm_watch"] = hist["watched_pct"].fillna(0.0) / 100.0
        hist["norm_delta_t"] = np.log1p(np.maximum(0.0, (split_ts - hist["timestamp"]) / 3600.0)) / 10.0

        self.user_seqs: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for uid, group in hist.groupby("user_id"):
            tail = group.tail(max_seq_len)
            self.user_seqs[uid] = (
                tail["item_idx"].to_numpy(dtype=np.int64),
                tail["genre_idx"].to_numpy(dtype=np.int64),
                tail["norm_watch"].to_numpy(dtype=np.float32),
                tail["norm_delta_t"].to_numpy(dtype=np.float32),
            )

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int):
        row = self.candidates.iloc[idx]
        uid = int(row["user_id"])
        target_item_idx = self.item_to_idx.get(int(row["item_id"]), 0)
        target_genre_idx = self.item_genre_idx.get(target_item_idx, 0)
        label = float(row["label"])

        seq_items = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_genres = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_watch = np.zeros(self.max_seq_len, dtype=np.float32)
        seq_dt = np.zeros(self.max_seq_len, dtype=np.float32)

        if uid in self.user_seqs:
            s_i, s_g, s_w, s_t = self.user_seqs[uid]
            length = len(s_i)
            seq_items[-length:] = s_i
            seq_genres[-length:] = s_g
            seq_watch[-length:] = s_w
            seq_dt[-length:] = s_t

        return (
            torch.tensor(seq_items),
            torch.tensor(seq_genres),
            torch.tensor(seq_watch),
            torch.tensor(seq_dt),
            torch.tensor(target_item_idx, dtype=torch.long),
            torch.tensor(target_genre_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


class TargetAwareHistoryTransformer(nn.Module):
    """
    Sequential Event History Model combining Self-Attention over recent user actions
    and Target-Aware Cross-Attention (DIN-style) conditioned on candidate item & time decay.
    """

    def __init__(self, num_items: int, num_genres: int, emb_dim: int = 32):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(num_genres + 1, emb_dim // 2, padding_idx=0)
        self.cont_proj = nn.Linear(2, emb_dim // 2)

        total_dim = emb_dim * 2
        self.query_proj = nn.Linear(emb_dim + emb_dim // 2, total_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_dim,
            nhead=4,
            dim_feedforward=total_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.seq_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, seq_items, seq_genres, seq_watch, seq_dt, target_item, target_genre):
        mask = seq_items == 0  # Padding mask [B, L]

        i_e = self.item_emb(seq_items)
        g_e = self.genre_emb(seq_genres)
        c_e = self.cont_proj(torch.stack([seq_watch, seq_dt], dim=-1))
        hist_repr = torch.cat([i_e, g_e, c_e], dim=-1)  # [B, L, D]

        # Contextualize sequence with Transformer layer
        all_padded = mask.all(dim=1)
        safe_mask = mask.clone()
        safe_mask[all_padded, -1] = False
        encoded_seq = self.seq_transformer(hist_repr, src_key_padding_mask=safe_mask)

        # Target query vector
        t_i = self.item_emb(target_item)
        t_g = self.genre_emb(target_genre)
        q = self.query_proj(torch.cat([t_i, t_g], dim=-1)).unsqueeze(1)  # [B, 1, D]

        # Target-aware scaled dot-product attention over history
        scores = torch.matmul(q, encoded_seq.transpose(1, 2)) / np.sqrt(encoded_seq.size(-1))
        scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_pooled = torch.matmul(attn_weights, encoded_seq).squeeze(1)  # [B, D]
        attn_pooled[all_padded] = 0.0

        # Mean pooling of valid sequence tokens
        valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        mean_pooled = (encoded_seq * (~mask).unsqueeze(-1)).sum(dim=1) / valid_counts

        q_flat = q.squeeze(1)
        combined = torch.cat([q_flat, attn_pooled, mean_pooled * q_flat], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


def train_and_predict_neural_history_model(
    splits: Dict,
    items: pd.DataFrame,
    max_seq_len: int = 30,
    epochs: int = 5,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Train TargetAwareHistoryTransformer on train split and generate predictions for all splits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_to_idx = {it: idx + 1 for idx, it in enumerate(items["item_id"].unique())}
    genre_to_idx = {g: idx + 1 for idx, g in enumerate(items["genre"].fillna("unk").unique())}

    datasets = {
        name: UserHistorySequenceDataset(
            hist_df, cand_df, items, item_to_idx, genre_to_idx, split_ts, max_seq_len=max_seq_len
        )
        for name, (hist_df, cand_df, split_ts) in splits.items()
    }

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    model = TargetAwareHistoryTransformer(len(item_to_idx), len(genre_to_idx), emb_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            s_i, s_g, s_w, s_t, t_i, t_g, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(s_i, s_g, s_w, s_t, t_i, t_g)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation split
        val_preds = predict_neural_dataset(model, datasets["val"], batch_size, device)
        val_auc = roc_auc_score(splits["val"][1]["label"], val_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        name: predict_neural_dataset(model, ds, batch_size, device)
        for name, ds in datasets.items()
    }


def predict_neural_dataset(
    model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            s_i, s_g, s_w, s_t, t_i, t_g, _ = [b.to(device) for b in batch]
            logits = model(s_i, s_g, s_w, s_t, t_i, t_g)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds, axis=0)


# =====================================================================
# 6. EVALUATION METRICS (CLASSIFICATION & RANKING)
# =====================================================================

def compute_ranking_metrics(df: pd.DataFrame, pred_col: str, k: int = 10) -> Dict[str, float]:
    """Compute global ROC-AUC, LogLoss, and per-user mean NDCG@K and Recall@K."""
    y_true = df["label"].to_numpy()
    y_pred = np.clip(df[pred_col].to_numpy(), 1e-7, 1.0 - 1e-7)

    roc_auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)

    ndcg_list = []
    recall_list = []

    ranked_df = df[["user_id", "label", pred_col]].sort_values(
        ["user_id", pred_col], ascending=[True, False]
    )

    for _, group in ranked_df.groupby("user_id", sort=False):
        labels = group["label"].to_numpy()
        total_pos = labels.sum()
        if total_pos == 0:
            continue

        top_k_labels = labels[:k]
        # Recall@K
        recall_list.append(top_k_labels.sum() / total_pos)

        # NDCG@K
        discounts = 1.0 / np.log2(np.arange(2, len(top_k_labels) + 2))
        dcg = np.sum(top_k_labels * discounts)
        ideal_len = int(min(total_pos, k))
        idcg = np.sum(np.ones(ideal_len) * discounts[:ideal_len])
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "ROC-AUC": round(float(roc_auc), 4),
        "LogLoss": round(float(logloss), 4),
        f"NDCG@{k}": round(float(np.mean(ndcg_list)), 4),
        f"Recall@{k}": round(float(np.mean(recall_list)), 4),
    }


# =====================================================================
# 7. MAIN EXPERIMENTAL PIPELINE
# =====================================================================

def main() -> None:
    set_random_seed(42)
    interactions, users, items = load_or_create_dataset(sample_users_count=2000, seed=42)
    print(f"Dataset summary: {len(users)} users, {len(items)} items, {len(interactions)} interactions.")

    splits = temporal_split_and_candidates(interactions, items, negatives_per_positive=4, seed=42)
    for s_name, (h_df, c_df, ts) in splits.items():
        print(f"Split [{s_name.upper()}]: history={len(h_df)} rows, candidates={len(c_df)} pairs, split_ts={ts}")

    # Step 1: Build Manual Baseline Features
    print("\nBuilding manual feature groups (User, Item, User-Item Direct & Indirect)...")
    manual_builder = ManualFeatureBuilder(users, items)
    feat_dfs = {
        name: manual_builder.build_features(h_df, c_df, ts)
        for name, (h_df, c_df, ts) in splits.items()
    }

    ignore_cols = {"user_id", "item_id", "label", "split_timestamp"}
    manual_feature_cols = [c for c in feat_dfs["train"].columns if c not in ignore_cols]
    print(f"Total manual features constructed: {len(manual_feature_cols)}")

    # Step 2: Train Baseline Gradient Boosting
    print("Training Baseline LightGBM on Manual Features...")
    baseline_gbdt = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample_for_bin=50000,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )
    baseline_gbdt.fit(
        feat_dfs["train"][manual_feature_cols],
        feat_dfs["train"]["label"],
        eval_set=[(feat_dfs["val"][manual_feature_cols], feat_dfs["val"]["label"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    # Step 3: Research Approach 1 - Automated SQL-like Feature Generation & Selection
    print("Running Automated Feature Synthesis & Validation Selection (Direction 1)...")
    auto_engine = AutomatedFeatureEngine(items, max_selected_features=18)
    for name, (h_df, _, ts) in splits.items():
        feat_dfs[name] = auto_engine.synthesize_candidates(feat_dfs[name], h_df, ts)

    selected_auto_cols = auto_engine.fit_select(feat_dfs["train"], feat_dfs["val"], manual_feature_cols)
    print(f"Selected top {len(selected_auto_cols)} automated features: {selected_auto_cols[:5]} ...")

    # Model A1: GBDT strictly on Automated Features + Raw Static Codes (Zero manual aggregations)
    static_codes = ["user_age_code", "user_income_code", "user_sex_code", "item_genre_code", "item_content_type_code", "release_year"]
    auto_only_cols = static_codes + selected_auto_cols
    gbdt_auto_only = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    gbdt_auto_only.fit(
        feat_dfs["train"][auto_only_cols],
        feat_dfs["train"]["label"],
        eval_set=[(feat_dfs["val"][auto_only_cols], feat_dfs["val"]["label"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    # Model A2: GBDT on Manual + Selected Automated Features
    combined_feat_cols = manual_feature_cols + selected_auto_cols
    gbdt_auto_plus_manual = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    gbdt_auto_plus_manual.fit(
        feat_dfs["train"][combined_feat_cols],
        feat_dfs["train"]["label"],
        eval_set=[(feat_dfs["val"][combined_feat_cols], feat_dfs["val"]["label"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    # Step 4: Research Approach 2 - Target-Aware Sequential Transformer (Direction 3)
    print("Training Target-Aware Event History Transformer (Direction 3)...")
    nn_preds = train_and_predict_neural_history_model(splits, items, max_seq_len=30, epochs=5, batch_size=256)
    for name in splits:
        feat_dfs[name]["nn_seq_prob"] = nn_preds[name]

    # Step 5: Hybrid GBDT + Neural Sequence Representation
    hybrid_cols = combined_feat_cols + ["nn_seq_prob"]
    gbdt_hybrid = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    gbdt_hybrid.fit(
        feat_dfs["train"][hybrid_cols],
        feat_dfs["train"]["label"],
        eval_set=[(feat_dfs["val"][hybrid_cols], feat_dfs["val"]["label"])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    # Step 6: Evaluate All Approaches on Held-Out Test Split
    test_eval = feat_dfs["test"].copy()
    test_eval["pred_baseline_manual"] = baseline_gbdt.predict_proba(test_eval[manual_feature_cols])[:, 1]
    test_eval["pred_auto_only"] = gbdt_auto_only.predict_proba(test_eval[auto_only_cols])[:, 1]
    test_eval["pred_manual_plus_auto"] = gbdt_auto_plus_manual.predict_proba(test_eval[combined_feat_cols])[:, 1]
    test_eval["pred_neural_seq"] = test_eval["nn_seq_prob"]
    test_eval["pred_hybrid_full"] = gbdt_hybrid.predict_proba(test_eval[hybrid_cols])[:, 1]

    models_to_compare = {
        "1. Baseline GBDT (Manual Features)": "pred_baseline_manual",
        "2. AutoFE GBDT (Zero Manual Aggs)": "pred_auto_only",
        "3. GBDT (Manual + AutoFE Selected)": "pred_manual_plus_auto",
        "4. Event History Transformer (Standalone)": "pred_neural_seq",
        "5. Hybrid GBDT (Manual + AutoFE + SeqNN)": "pred_hybrid_full",
    }

    results = {}
    for label_name, col in models_to_compare.items():
        results[label_name] = compute_ranking_metrics(test_eval, col, k=10)

    results_table = pd.DataFrame(results).T
    print("\n================ FINAL TEST SET EVALUATION ================")
    print(results_table.to_string())


if __name__ == "__main__":
    main()