"""
Saf numpy / pandas / scipy yardımcıları. fastapi veya statsmodels gerektirmez, bu yüzden tek başına test edilebilir.
"""
import csv
import io
import math
import os
import re
import tempfile
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------- biçimlendirme
def fmt_p(p: float) -> str:
    try:
        if p is None or pd.isna(p):
            return "-"
    except (TypeError, ValueError):
        return "-"
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def clean_json(o: Any) -> Any:
    """numpy tiplerini Python tiplerine çevirir; NaN/inf -> None (JSON uyumlu)."""
    if o is None or o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean_json(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, str):
        return o
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def clean_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in tables:
        rows = clean_json(t.get("rows", []))
        rows = [{k: ("-" if v is None else v) for k, v in r.items()} for r in rows]
        out.append({"title": str(t.get("title", "")), "rows": rows})
    return out


# --------------------------------------------------------------- R kodu yardımcıları
def rn(name: str) -> str:
    """R değişken adı: gerekirse `backtick` ile korur."""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9._]*|\.[A-Za-z._][A-Za-z0-9._]*", name):
        return name
    return "`" + name.replace("`", "\\`") + "`"


def rq(name: str) -> str:
    """R string literal."""
    return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --------------------------------------------------------------- harflendirme
def compute_compact_letters_from_pairs(means_dict: Dict[str, float], non_sig_pairs: List[tuple]) -> Dict[str, str]:
    """
    Compact letter display. Anlamlı fark OLMAYAN her çift ortak bir harf paylaşır,
    anlamlı farkı olan hiçbir çift harf paylaşmaz. Bunun için 'anlamsız' grafın
    maksimal klikleri (Bron-Kerbosch) bulunur; her klik bir harf olur.
    """
    groups = sorted(means_dict.keys(), key=lambda g: means_dict[g], reverse=True)
    adj = {g: set() for g in groups}
    for a, b in non_sig_pairs:
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)

    cliques: List[set] = []

    def bron_kerbosch(r: set, p: set, x: set) -> None:
        if not p and not x:
            cliques.append(set(r))
            return
        pivot = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[pivot]):
            bron_kerbosch(r | {v}, p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    bron_kerbosch(set(), set(groups), set())

    rank = {g: i for i, g in enumerate(groups)}
    cliques.sort(key=lambda c: sorted(rank[g] for g in c))

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters = {g: "" for g in groups}
    for idx, clq in enumerate(cliques):
        ch = alphabet[idx] if idx < len(alphabet) else f"({idx + 1})"
        for g in sorted(clq, key=lambda g: rank[g]):
            letters[g] += ch
    return letters


# --------------------------------------------------------------- post-hoc
def tukey_pairs(sub_df: pd.DataFrame, val_col: str, grp_col: str, alpha: float):
    """Tukey HSD (dengesiz n için Tukey-Kramer). scipy.stats.tukey_hsd: yuvarlanmamış p ve GA verir."""
    key = sub_df[grp_col].astype(str)
    groups = sorted(key.unique())
    samples = [sub_df.loc[key == g, val_col].astype(float).values for g in groups]
    res = stats.tukey_hsd(*samples)
    ci = res.confidence_interval(confidence_level=1 - alpha)
    rows, non_sig = [], []
    for i, j in combinations(range(len(groups)), 2):
        diff = float(res.statistic[i, j])  # ortalama_i - ortalama_j
        p_val = float(res.pvalue[i, j])
        reject = p_val < alpha
        if not reject:
            non_sig.append((groups[i], groups[j]))
        rows.append({
            "Grup 1": groups[i],
            "Grup 2": groups[j],
            "Ort. Farkı": round(diff, 4),
            "p-adj": fmt_p(p_val),
            f"Alt (%{(1 - alpha) * 100:g})": round(float(ci.low[i, j]), 4),
            f"Üst (%{(1 - alpha) * 100:g})": round(float(ci.high[i, j]), 4),
            "Anlamlı": "Evet *" if reject else "Hayır",
        })
    return rows, non_sig


def games_howell_pairs(sub_df: pd.DataFrame, val_col: str, grp_col: str, alpha: float):
    key = sub_df[grp_col].astype(str)
    groups = sorted(key.unique())
    k = len(groups)
    st = {}
    for g in groups:
        arr = sub_df.loc[key == g, val_col].astype(float).values
        st[g] = {"n": len(arr), "m": float(np.mean(arr)), "v": float(np.var(arr, ddof=1))}

    rows, non_sig = [], []
    for g1, g2 in combinations(groups, 2):
        s1, s2 = st[g1], st[g2]
        a1, a2 = s1["v"] / s1["n"], s2["v"] / s2["n"]
        se2 = max(a1 + a2, 1e-18)
        se = math.sqrt(se2)
        df_w = se2 ** 2 / ((a1 ** 2) / max(1, s1["n"] - 1) + (a2 ** 2) / max(1, s2["n"] - 1))
        diff = s1["m"] - s2["m"]
        q_stat = abs(diff) / se * math.sqrt(2)  # q = sqrt(2) * t
        p_val = float(stats.studentized_range.sf(q_stat, k, df_w))
        q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df_w)) / math.sqrt(2)
        reject = p_val < alpha
        if not reject:
            non_sig.append((g1, g2))
        rows.append({
            "Grup 1": g1,
            "Grup 2": g2,
            "Ort. Farkı": round(diff, 4),
            "p-adj": fmt_p(p_val),
            f"Alt (%{(1 - alpha) * 100:g})": round(diff - q_crit * se, 4),
            f"Üst (%{(1 - alpha) * 100:g})": round(diff + q_crit * se, 4),
            "Anlamlı": "Evet *" if reject else "Hayır",
        })
    return rows, non_sig


def welch_anova(samples: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """Welch's ANOVA -> (F, df1, df2, p). Varyanslar eşit olmadığında klasik F yerine kullanılır."""
    k = len(samples)
    n = np.array([len(s) for s in samples], dtype=float)
    m = np.array([np.mean(s) for s in samples], dtype=float)
    v = np.maximum(np.array([np.var(s, ddof=1) for s in samples], dtype=float), 1e-18)
    w = n / v
    w_sum = w.sum()
    m_w = (w * m).sum() / w_sum
    a = (w * (m - m_w) ** 2).sum() / (k - 1)
    lam = (((1 - w / w_sum) ** 2) / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    f_val = a / b
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * lam)
    return float(f_val), float(df1), float(df2), float(stats.f.sf(f_val, df1, df2))


# --------------------------------------------------------------- dosya okuma
def _decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "cp1254", "latin-1"):  # cp1254 = Türkçe Windows (Excel'in "CSV" kaydı)
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


def _sniff_sep(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            counts = {s: line.count(s) for s in (",", ";", "\t", "|")}
            best = max(counts, key=counts.get)
            return best if counts[best] > 0 else ","
    return ","


def _fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    """'12,5' gibi Türkçe ondalıklı metin sütunlarını sayıya çevirir (yalnızca TÜM dolu değerler bu kalıptaysa)."""
    pattern = re.compile(r"^-?\d+([.,]\d+)?$")
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        nn = s.dropna().astype(str).str.strip()
        if nn.empty or not nn.str.contains(",", regex=False).any():
            continue
        if not nn.map(lambda v: bool(pattern.match(v))).all():
            continue
        full = pd.Series(np.nan, index=s.index, dtype=float)
        full.loc[nn.index] = pd.to_numeric(nn.str.replace(",", ".", regex=False)).values
        df[col] = full
    return df


def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    cols = []
    for c in df.columns:
        name = str(c).strip().replace('"', "'") or "Sutun"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        cols.append(name)
    df.columns = cols
    return df


def read_table_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """csv / xlsx / xls / sav okur. Hata durumunda ValueError fırlatır."""
    name = (filename or "").lower()
    if name.endswith(".csv") or name.endswith(".txt") or name.endswith(".tsv"):
        text = _decode(content)
        df = pd.read_csv(io.StringIO(text), sep=_sniff_sep(text))
        df = _fix_decimal_commas(df)
    elif name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    elif name.endswith(".sav"):
        import pyreadstat  # path ister, BytesIO kabul etmez
        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
            tmp.write(content)
            path = tmp.name
        try:
            df, _ = pyreadstat.read_sav(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    else:
        raise ValueError("Desteklenmeyen format (.csv, .xlsx, .xls, .sav kullanın).")
    if df.empty or df.shape[1] == 0:
        raise ValueError("Dosya boş veya okunabilir sütun içermiyor.")
    return _tidy_columns(df)
