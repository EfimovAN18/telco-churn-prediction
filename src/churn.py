"""
Подготовка данных, модели и бизнес-метрики для прогноза оттока клиентов.

Используется в ноутбуке notebooks/churn_analysis.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"
TARGET = "churn"
SEED = 42

NUMERIC = ["tenure", "monthly_charges", "total_charges", "services_count"]
BINARY = ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]
CATEGORICAL = [
    "gender", "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "contract", "payment_method",
]
ADDON_SERVICES = ["online_security", "online_backup", "device_protection",
                  "tech_support", "streaming_tv", "streaming_movies"]


def to_snake(name: str) -> str:
    """CamelCase -> snake_case: MonthlyCharges -> monthly_charges."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper() or (i + 1 < len(name) and name[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path.name}. Положите файл в data/raw/ (см. data/raw/README.md).")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - имена столбцов в snake_case;
    - TotalCharges хранится строкой: у 11 новых клиентов (tenure = 0) там пробел -> 0, они ещё не платили;
    - Yes/No -> 1/0, целевая переменная -> 0/1.
    """
    out = df.rename(columns={c: to_snake(c) for c in df.columns})
    out["total_charges"] = pd.to_numeric(out["total_charges"].str.strip().replace("", "0"))
    for col in ["partner", "dependents", "phone_service", "paperless_billing", TARGET]:
        out[col] = out[col].map({"Yes": 1, "No": 0}).astype(int)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Число подключённых доп. услуг и группы по сроку жизни клиента."""
    out = df.copy()
    out["services_count"] = (out[ADDON_SERVICES] == "Yes").sum(axis=1)
    out["tenure_group"] = pd.cut(out["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                 labels=["0–6 мес", "7–12 мес", "1–2 года", "2–4 года", "4–6 лет"])
    return out


def make_preprocessor() -> ColumnTransformer:
    """Масштабирование чисел и one-hot для категорий — общий для всех моделей."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"), CATEGORICAL),
    ])


def profit_curve(y_true: np.ndarray, proba: np.ndarray, customer_value: np.ndarray,
                 offer_cost: float, success_rate: float) -> pd.DataFrame:
    """
    Прибыль кампании удержания при разных порогах.
    Клиенту с вероятностью оттока выше порога предлагаем скидку стоимостью offer_cost.
    Если клиент действительно собирался уйти, предложение удерживает его с вероятностью success_rate,
    и мы сохраняем его ценность customer_value. Предложение получают и те, кто не ушёл бы, — это чистые затраты.
    """
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        target = proba >= t
        saved = (success_rate * customer_value * (target & (y_true == 1))).sum()
        cost = offer_cost * target.sum()
        rows.append({"порог": t, "клиентов в кампании": int(target.sum()),
                     "сохранённая выручка": saved, "затраты": cost, "прибыль": saved - cost})
    return pd.DataFrame(rows)
