import ast
import io
import math
import operator
from itertools import combinations
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from statsmodels.duration.hazard_regression import PHReg

app = FastAPI(title="Kodsuz Akademik İstatistik & Veri Analizi — Pro 24")
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSION_STORE: Dict[str, pd.DataFrame] = {}
REPORT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "-"
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


class AnalysisRequest(BaseModel):
    session_id: str = "default"
    test_type: str
    dependent_vars: List[str]
    group_var: Optional[str] = None
    second_factor: Optional[str] = None
    alpha: float = 0.05
    effect_size: float = 0.5
    power_target: float = 0.80
    mu0: float = 0.0


class CellUpdateRequest(BaseModel):
    session_id: str = "default"
    row_index: int
    column: str
    value: str


class ComputeColumnRequest(BaseModel):
    session_id: str = "default"
    new_col_name: str
    expression: str


# Güvenli AST Matematik İfadeleri Değerlendirici (eval yerine)
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
SAFE_FUNCS = {
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "round": np.round,
    "zscore": lambda s: (s - np.nanmean(s)) / np.nanstd(s, ddof=1),
}


def safe_eval_expr(node, env: Dict[str, Any]):
    if isinstance(node, ast.Expression):
        return safe_eval_expr(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"Bilinmeyen değişken: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](safe_eval_expr(node.left, env), safe_eval_expr(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](safe_eval_expr(node.operand, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCS:
        args = [safe_eval_expr(a, env) for a in node.args]
        return SAFE_FUNCS[node.func.id](*args)
    raise ValueError("Desteklenmeyen veya güvenli olmayan ifade.")


def compute_compact_letters_from_pairs(means_dict: Dict[str, float], non_sig_pairs: List[tuple]) -> Dict[str, str]:
    groups = sorted(means_dict.keys(), key=lambda g: means_dict[g], reverse=True)
    n = len(groups)
    non_sig = {g: set([g]) for g in groups}

    for g1, g2 in non_sig_pairs:
        if g1 in non_sig and g2 in non_sig:
            non_sig[g1].add(g2)
            non_sig[g2].add(g1)

    cliques = []
    for i in range(n):
        cand = {groups[i]}
        for j in range(i + 1, n):
            if all(groups[j] in non_sig[member] for member in cand):
                cand.add(groups[j])
        if not any(cand.issubset(existing) for existing in cliques):
            cliques.append(cand)

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    letters_map = {g: "" for g in groups}
    for idx, clq in enumerate(cliques):
        ch = alphabet[idx] if idx < len(alphabet) else f"g{idx}"
        for g in clq:
            letters_map[g] += ch
    return letters_map


def games_howell_pairs(sub_df: pd.DataFrame, val_col: str, grp_col: str, alpha: float):
    groups = sorted(sub_df[grp_col].astype(str).unique())
    k = len(groups)
    stats_map = {}
    for g in groups:
        arr = sub_df[sub_df[grp_col].astype(str) == g][val_col].values
        stats_map[g] = {"n": len(arr), "m": np.mean(arr), "v": np.var(arr, ddof=1)}

    rows = []
    non_sig_pairs = []
    for g1, g2 in combinations(groups, 2):
        s1, s2 = stats_map[g1], stats_map[g2]
        se2 = (s1["v"] / s1["n"]) + (s2["v"] / s2["n"])
        se = math.sqrt(se2) if se2 > 0 else 1e-9
        df_w = (se2 ** 2) / (((s1["v"] / s1["n"]) ** 2) / max(1, s1["n"] - 1) + ((s2["v"] / s2["n"]) ** 2) / max(1, s2["n"] - 1))
        diff = s1["m"] - s2["m"]
        q_stat = abs(diff) / se * math.sqrt(2)
        p_val = float(stats.studentized_range.sf(q_stat, k, df_w))
        q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df_w)) / math.sqrt(2)
        reject = p_val < alpha
        if not reject:
            non_sig_pairs.append((g1, g2))
        rows.append({
            "Grup 1": g1,
            "Grup 2": g2,
            "Ort. Farkı": round(float(diff), 4),
            "p-adj": fmt_p(p_val),
            "Alt (%95)": round(float(diff - q_crit * se), 4),
            "Üst (%95)": round(float(diff + q_crit * se), 4),
            "Anlamlı": "Evet *" if reject else "Hayır"
        })
    return rows, non_sig_pairs


def build_dataset_response(df: pd.DataFrame, filename: str, session_id: str):
    SESSION_STORE[session_id] = df
    columns_meta = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]) and unique_count > 10:
            var_type = "Sürekli"
        elif unique_count <= 10:
            var_type = "Kategorik"
        else:
            var_type = "Nominal"

        columns_meta.append({
            "name": col,
            "dtype": dtype,
            "var_type": var_type,
            "missing": int(df[col].isna().sum()),
            "unique": int(unique_count)
        })

    preview = df.replace({np.nan: ""}).to_dict(orient="records")
    return {
        "session_id": session_id,
        "filename": filename,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": columns_meta,
        "preview": preview
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/raw-csv")
async def get_raw_csv(session_id: str = "default"):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Aktif veri seti bulunamadı.")
    return {"csv": SESSION_STORE[session_id].to_csv(index=False)}


@app.post("/api/clear-history")
async def clear_history(session_id: str = "default"):
    REPORT_HISTORY[session_id] = []
    return {"status": "cleared"}


@app.post("/api/demo")
async def load_demo_dataset(session_id: str = "default"):
    np.random.seed(42)
    n = 60
    gruplar = np.repeat(["Kontrol", "Doz_1", "Doz_2"], 20)
    blok = np.tile(["Blok_A", "Blok_B"], 30)
    kovaryant = np.random.normal(25, 3.0, n)
    param_1 = np.concatenate([
        np.random.normal(42, 4.5, 20),
        np.random.normal(54, 5.0, 20),
        np.random.normal(63, 4.8, 20)
    ]) + 0.4 * kovaryant
    param_2 = param_1 * 0.65 + np.random.normal(10, 3.2, n)
    param_3 = param_1 * 0.80 + np.random.normal(5, 2.8, n)
    biomarker = param_1 * 0.12 + np.random.normal(2.5, 0.8, n)
    durum_bin = (param_1 > 62).astype(int)
    durum_sonra = (param_1 > 58).astype(int)
    sure_gun = np.random.randint(10, 90, n)
    madde_1 = np.random.randint(2, 6, n)
    madde_2 = np.clip(madde_1 + np.random.randint(-1, 2, n), 1, 5)
    madde_3 = np.clip(madde_1 + np.random.randint(-1, 2, n), 1, 5)
    madde_4 = np.clip(madde_1 + np.random.randint(-1, 2, n), 1, 5)

    df = pd.DataFrame({
        "Uygulama_Grubu": gruplar,
        "Alt_Faktor": blok,
        "Kovaryant_Yas": np.round(kovaryant, 1),
        "Olcum_T1": np.round(param_1, 2),
        "Olcum_T2": np.round(param_2, 2),
        "Olcum_T3": np.round(param_3, 2),
        "Biyobelirtec": np.round(biomarker, 3),
        "Sure_Gun": sure_gun,
        "Olay_Durumu": durum_bin,
        "Olay_Sonra": durum_sonra,
        "Anket_M1": madde_1,
        "Anket_M2": madde_2,
        "Anket_M3": madde_3,
        "Anket_M4": madde_4
    })
    return build_dataset_response(df, "Demo_Kapsamli_VeriSeti.xlsx", session_id)


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), session_id: str = "default"):
    content = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        elif filename.endswith(".sav"):
            import pyreadstat
            df, _ = pyreadstat.read_sav(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Desteklenmeyen format (.csv, .xlsx, .sav kullanın).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okuma hatası: {str(e)}")

    return build_dataset_response(df, file.filename, session_id)


@app.post("/api/update-cell")
async def update_cell(req: CellUpdateRequest):
    if req.session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Aktif veri seti bulunamadı.")
    df = SESSION_STORE[req.session_id]
    if req.column not in df.columns or req.row_index < 0 or req.row_index >= len(df):
        raise HTTPException(status_code=400, detail="Geçersiz satır veya sütun.")

    val_str = req.value.strip()
    if val_str == "":
        df.at[req.row_index, req.column] = np.nan
    else:
        try:
            num_val = float(val_str)
            df.at[req.row_index, req.column] = int(num_val) if num_val.is_integer() and pd.api.types.is_integer_dtype(df[req.column]) else num_val
        except ValueError:
            df.at[req.row_index, req.column] = val_str

    SESSION_STORE[req.session_id] = df
    return {"status": "ok"}


@app.post("/api/compute-column")
async def compute_column(req: ComputeColumnRequest):
    if req.session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Aktif veri seti bulunamadı.")
    df = SESSION_STORE[req.session_id].copy()
    col_name = req.new_col_name.strip()
    expr = req.expression.strip()

    if not col_name or not expr:
        raise HTTPException(status_code=400, detail="Yeni sütun adı ve formül boş olamaz.")

    try:
        local_env = {c: pd.to_numeric(df[c], errors="coerce").values for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
        tree = ast.parse(expr, mode="eval")
        result_arr = safe_eval_expr(tree, local_env)
        df[col_name] = np.round(result_arr, 4) if isinstance(result_arr, np.ndarray) else result_arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Formül hatası: {str(e)}")

    return build_dataset_response(df, "Guncellenmis_VeriSeti.xlsx", req.session_id)


@app.post("/api/analyze")
async def run_analysis(req: AnalysisRequest):
    REPORT_HISTORY.setdefault(req.session_id, [])

    # GÜÇ ANALİZİ (VERİ GEREKTİRMEZ)
    if req.test_type == "power":
        power_t = TTestIndPower()
        n_per_group = power_t.solve_power(effect_size=req.effect_size, alpha=req.alpha, power=req.power_target, alternative="two-sided")
        n_ceil = math.ceil(float(n_per_group))

        power_f = FTestAnovaPower()
        n_anova = power_f.solve_power(effect_size=req.effect_size, ngroups=3, alpha=req.alpha, power=req.power_target)
        n_anova_ceil = math.ceil(float(n_anova))

        tables = [{
            "title": f"A Priori Örneklem Büyüklüğü ve Güç Analizi (α = {req.alpha}, Güç = {req.power_target}, Etki = {req.effect_size})",
            "rows": [
                {"Test Tasarımı": "Bağımsız İki Örneklem T-Testi", "Grup Başına N": n_ceil, "Toplam Örneklem (N)": n_ceil * 2, "Hedef Güç (1-β)": req.power_target},
                {"Test Tasarımı": "Tek Yönlü ANOVA (3 Grup)", "Grup Başına N": math.ceil(n_anova_ceil / 3), "Toplam Örneklem (N)": n_anova_ceil, "Hedef Güç (1-β)": req.power_target}
            ]
        }]
        apa_note = f"α = {req.alpha} anlamlılık düzeyi ve {req.effect_size} etki büyüklüğünde {req.power_target} istatistiksel güce ulaşmak için bağımsız iki grup karşılaştırmasında grup başına en az {n_ceil} (toplam {n_ceil * 2}) örneklem gerekmektedir."
        r_code = f"power.t.test(delta = {req.effect_size}, sd = 1, sig.level = {req.alpha}, power = {req.power_target}, type = 'two.sample')"
        REPORT_HISTORY[req.session_id].append({"title": "Güç Analizi (Sample Size)", "tables": tables, "r_code": r_code, "apa_note": apa_note, "notes": []})
        return {"tables": tables, "r_code": r_code, "chart": None, "apa_note": apa_note, "notes": []}

    if req.session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Önce bir veri seti yükleyin veya Demo Veri butonuna basın.")

    df = SESSION_STORE[req.session_id].copy()
    tables = []
    notes = []
    r_code = ""
    chart_data = None
    apa_note = ""
    report_title = req.test_type.upper()

    # 1. BETİMSEL İSTATİSTİKLER + NORMALLİK
    if req.test_type == "descriptive":
        report_title = "Betimsel İstatistikler & Normallik"
        if not req.dependent_vars:
            raise HTTPException(status_code=400, detail="En az 1 sayısal değişken seçin.")
        rows = []
        for var in req.dependent_vars:
            series = pd.to_numeric(df[var], errors="coerce").dropna()
            if len(series) < 3:
                raise HTTPException(status_code=400, detail=f"'{var}' sütunu yeterli sayısal veri içermiyor.")
            shapiro_stat, shapiro_p = stats.shapiro(series)
            mean_val = float(series.mean())
            sd_val = float(series.std(ddof=1))
            se_val = float(stats.sem(series))
            cv_val = (sd_val / mean_val * 100) if mean_val != 0 else 0.0

            rows.append({
                "Değişken": var,
                "N": int(len(series)),
                "Ortalama": round(mean_val, 3),
                "Std. Sapma (SD)": round(sd_val, 3),
                "Std. Hata (SE)": round(se_val, 3),
                "%CV": round(cv_val, 2),
                "Medyan": round(float(series.median()), 3),
                "IQR": round(float(series.quantile(0.75) - series.quantile(0.25)), 3),
                "Min": round(float(series.min()), 3),
                "Maks": round(float(series.max()), 3),
                "Shapiro W": round(float(shapiro_stat), 4),
                "Normallik (p)": f"{fmt_p(shapiro_p)}" + (" (İhlal *)" if shapiro_p < 0.05 else " (Normal)")
            })
            if shapiro_p < 0.05:
                notes.append(f"'{var}' normal dağılımdan sapmaktadır (p {fmt_p(shapiro_p)}); parametrik olmayan testleri tercih edin.")

        tables.append({"title": "Betimsel İstatistikler ve Shapiro-Wilk Normallik Testi", "rows": rows})
        r0 = rows[0]
        apa_note = f"İncelenen {r0['Değişken']} değişkenine ait ortalama değer {r0['Ortalama']} ± {r0['Std. Sapma (SD)']} (Medyan = {r0['Medyan']}, IQR = {r0['IQR']}) olarak hesaplanmıştır (Shapiro-Wilk W = {r0['Shapiro W']})."
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"vars <- c({vars_str})\nsummary(df[, vars])\nlapply(df[, vars], shapiro.test)"

    # 2. TEK ÖRNEKLEM T-TESTİ
    elif req.test_type == "ttest1":
        report_title = "Tek Örneklem T-Testi"
        if not req.dependent_vars:
            raise HTTPException(status_code=400, detail="En az 1 sayısal değişken seçin.")
        rows = []
        for var in req.dependent_vars:
            series = pd.to_numeric(df[var], errors="coerce").dropna()
            t_stat, p_val = stats.ttest_1samp(series, popmean=req.mu0)
            cohen_d = (series.mean() - req.mu0) / series.std(ddof=1) if series.std(ddof=1) != 0 else 0.0
            rows.append({
                "Değişken": var,
                "N": len(series),
                "Ortalama": round(float(series.mean()), 3),
                "SD": round(float(series.std(ddof=1)), 3),
                "Test Değeri (μ₀)": req.mu0,
                "t": round(float(t_stat), 4),
                "df": len(series) - 1,
                "p": fmt_p(float(p_val)),
                "Cohen's d": round(float(cohen_d), 3)
            })
        tables.append({"title": f"Tek Örneklem T-Testi (Test Değeri = {req.mu0})", "rows": rows})
        r0 = rows[0]
        apa_note = f"Tek örneklem t-testi sonucuna göre {r0['Değişken']} ortalaması ({r0['Ortalama']}) test değeri olan {req.mu0}'dan t({r0['df']}) = {r0['t']}, p {r0['p']}, d = {r0['Cohen\'s d']} düzeyinde farklılaşmaktadır."
        r_code = f"t.test(df[['{req.dependent_vars[0]}']], mu = {req.mu0})"

    # 3. BAĞIMSIZ İKİ ÖRNEKLEM T-TESTİ
    elif req.test_type == "ttest_ind":
        report_title = "Bağımsız İki Örneklem T-Testi"
        if not req.group_var or not req.dependent_vars:
            raise HTTPException(status_code=400, detail="Bağımlı değişken ve gruplama değişkeni seçin.")
        groups = df[req.group_var].dropna().unique()
        if len(groups) != 2:
            raise HTTPException(status_code=400, detail=f"Gruplama değişkeni tam 2 kategori içermelidir (Seçilende {len(groups)} kategori var).")

        rows, group_summary = [], []
        for var in req.dependent_vars:
            g1 = pd.to_numeric(df[df[req.group_var] == groups[0]][var], errors="coerce").dropna()
            g2 = pd.to_numeric(df[df[req.group_var] == groups[1]][var], errors="coerce").dropna()
            levene_stat, levene_p = stats.levene(g1, g2, center="median")
            equal_var = levene_p >= 0.05
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
            n1, n2 = len(g1), len(g2)
            v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
            if equal_var:
                df_val = float(n1 + n2 - 2)
            else:
                df_val = ((v1 / n1 + v2 / n2) ** 2) / (((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1))
            pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            cohen_d = (g1.mean() - g2.mean()) / pooled_sd if pooled_sd != 0 else 0.0

            if levene_p < 0.05:
                notes.append(f"'{var}' için Levene varyans homojenliği ihlal edildi (p {fmt_p(levene_p)}); otomatik olarak Welch's t-testi uygulandı.")

            rows.append({
                "Değişken": var,
                "Yöntem": "Student's t" if equal_var else "Welch's t",
                "t": round(float(t_stat), 4),
                "df": round(df_val, 1) if not equal_var else int(df_val),
                "p": fmt_p(p_val),
                "Cohen's d": round(float(cohen_d), 3),
                "Levene p": fmt_p(levene_p)
            })
            group_summary.extend([
                {"Değişken": var, "Grup": str(groups[0]), "N": n1, "Ortalama": round(float(g1.mean()), 3), "SD": round(float(g1.std(ddof=1)), 3), "SE": round(float(stats.sem(g1)), 3)},
                {"Değişken": var, "Grup": str(groups[1]), "N": n2, "Ortalama": round(float(g2.mean()), 3), "SD": round(float(g2.std(ddof=1)), 3), "SE": round(float(stats.sem(g2)), 3)},
            ])

        tables.append({"title": "Bağımsız Örneklem T-Testi Sonuçları", "rows": rows})
        tables.append({"title": "Grup Tanımlayıcı İstatistikleri", "rows": group_summary})
        r0 = rows[0]
        apa_note = f"Bağımsız örneklem t-testi ({r0['Yöntem']}) sonucuna göre {groups[0]} (Ort. = {group_summary[0]['Ortalama']}, SD = {group_summary[0]['SD']}) ile {groups[1]} (Ort. = {group_summary[1]['Ortalama']}, SD = {group_summary[1]['SD']}) grupları arasında {r0['Değişken']} bakımından t({r0['df']}) = {r0['t']}, p {r0['p']}, d = {r0['Cohen\'s d']} elde edilmiştir."
        chart_data = {
            "type": "bar_error",
            "title": f"{req.group_var} Gruplarına Göre {req.dependent_vars[0]} (±SE)",
            "labels": [str(groups[0]), str(groups[1])],
            "values": [group_summary[0]["Ortalama"], group_summary[1]["Ortalama"]],
            "errors": [group_summary[0]["SE"], group_summary[1]["SE"]],
            "y_label": req.dependent_vars[0]
        }
        r_code = f"t.test({req.dependent_vars[0]} ~ {req.group_var}, data = df)"

    # 4. EŞLEŞTİRİLMİŞ T-TESTİ
    elif req.test_type == "ttest_paired":
        report_title = "Eşleştirilmiş Örneklem T-Testi"
        if len(req.dependent_vars) != 2:
            raise HTTPException(status_code=400, detail="Eşleştirilmiş T-Testi için tam 2 ölçüm değişkeni seçin.")
        v1, v2 = req.dependent_vars[0], req.dependent_vars[1]
        sub = df[[v1, v2]].apply(pd.to_numeric, errors="coerce").dropna()
        t_stat, p_val = stats.ttest_rel(sub[v1], sub[v2])
        diff = sub[v1] - sub[v2]
        cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else 0.0
        _, sh_p = stats.shapiro(diff)
        if sh_p < 0.05:
            notes.append(f"Fark skorları normal dağılımdan sapmaktadır (Shapiro p {fmt_p(sh_p)}); Wilcoxon İşaretli Sıralar testini de kontrol edin.")

        tables.append({
            "title": "Eşleştirilmiş (Bağımlı) Örneklem T-Testi",
            "rows": [{
                "Karşılaştırma": f"{v1} - {v2}",
                "N": len(sub),
                "Ort. Fark": round(float(diff.mean()), 4),
                "SD Fark": round(float(diff.std(ddof=1)), 4),
                "t": round(float(t_stat), 4),
                "df": len(sub) - 1,
                "p": fmt_p(p_val),
                "Cohen's dz": round(float(cohen_d), 3)
            }]
        })
        apa_note = f"{v1} ve {v2} ölçümleri arasındaki eşleştirilmiş t-testi analizinde ortalama fark {diff.mean():.3f} (SD = {diff.std(ddof=1):.3f}) olarak bulunmuştur, t({len(sub)-1}) = {t_stat:.3f}, p {fmt_p(p_val)}."
        r_code = f"t.test(df[['{v1}']], df[['{v2}']], paired = TRUE)"

    # 5. WILCOXON İŞARETLİ SIRALAR TESTİ
    elif req.test_type == "wilcoxon":
        report_title = "Wilcoxon İşaretli Sıralar Testi"
        if len(req.dependent_vars) != 2:
            raise HTTPException(status_code=400, detail="Wilcoxon testi için tam 2 ölçüm değişkeni seçin.")
        v1, v2 = req.dependent_vars[0], req.dependent_vars[1]
        sub = df[[v1, v2]].apply(pd.to_numeric, errors="coerce").dropna()
        w_stat, p_val = stats.wilcoxon(sub[v1], sub[v2])
        tables.append({
            "title": "Wilcoxon İşaretli Sıralar (Eşleştirilmiş Non-Parametrik) Testi",
            "rows": [{
                "Karşılaştırma": f"{v1} vs {v2}",
                "N": len(sub),
                f"Medyan ({v1})": round(float(sub[v1].median()), 3),
                f"Medyan ({v2})": round(float(sub[v2].median()), 3),
                "Wilcoxon W": round(float(w_stat), 4),
                "p": fmt_p(float(p_val))
            }]
        })
        apa_note = f"Wilcoxon işaretli sıralar testi sonucunda {v1} (Medyan = {sub[v1].median():.2f}) ile {v2} (Medyan = {sub[v2].median():.2f}) arasında W = {w_stat:.2f}, p {fmt_p(float(p_val))} bulunmuştur."
        r_code = f"wilcox.test(df[['{v1}']], df[['{v2}']], paired = TRUE)"

    # 6. FRIEDMAN TESTİ
    elif req.test_type == "friedman":
        report_title = "Friedman Testi (Tekrarlı Non-Parametrik)"
        if len(req.dependent_vars) < 3:
            raise HTTPException(status_code=400, detail="Friedman testi için en az 3 tekrarlı ölçüm değişkeni seçin.")
        sub = df[req.dependent_vars].apply(pd.to_numeric, errors="coerce").dropna()
        stat, p_val = stats.friedmanchisquare(*[sub[c].values for c in req.dependent_vars])
        n, k = len(sub), len(req.dependent_vars)
        kendall_w = stat / (n * (k - 1))
        tables.append({
            "title": "Friedman Testi Sonuçları",
            "rows": [{
                "N": n,
                "Koşul Sayısı (k)": k,
                "Ki-Kare (χ²)": round(float(stat), 4),
                "df": k - 1,
                "p": fmt_p(float(p_val)),
                "Kendall's W": round(float(kendall_w), 3)
            }]
        })
        apa_note = f"Friedman testi sonucunda {k} tekrarlı ölçüm arasında χ²({k-1}) = {stat:.3f}, p {fmt_p(float(p_val))}, Kendall's W = {kendall_w:.3f} elde edilmiştir."
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"friedman.test(as.matrix(na.omit(df[, c({vars_str})])))"

    # 7. TEK YÖNLÜ ANOVA + LEVENE + TUKEY / GAMES-HOWELL + HARFLENDİRME
    elif req.test_type == "anova":
        report_title = "Tek Yönlü ANOVA & Harflendirme"
        if not req.group_var or not req.dependent_vars:
            raise HTTPException(status_code=400, detail="Bağımlı değişken ve faktör değişkeni seçin.")

        var = req.dependent_vars[0]
        sub_df = df[[var, req.group_var]].dropna()
        sub_df[var] = pd.to_numeric(sub_df[var], errors="coerce")
        sub_df = sub_df.dropna()

        model = ols(f'Q("{var}") ~ C(Q("{req.group_var}"))', data=sub_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        total_ss = float(anova_table["sum_sq"].sum())
        msw = float(anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"])

        anova_rows = []
        f_val_main, p_val_main, df1_main, df2_main, eta_sq, omega_sq = 0, 1, 0, 0, 0, 0
        for idx, row in anova_table.iterrows():
            is_factor = "C(" in str(idx)
            label = req.group_var if is_factor else "Hata (Residual)"
            ss_val = float(row["sum_sq"])
            if is_factor:
                f_val_main = round(float(row["F"]), 4)
                p_val_main = float(row["PR(>F)"])
                df1_main = int(row["df"])
                eta_sq = round(ss_val / total_ss, 3) if total_ss > 0 else 0
                omega_sq = round(max(0.0, (ss_val - df1_main * msw) / (total_ss + msw)), 3)
            else:
                df2_main = int(row["df"])
            anova_rows.append({
                "Varyasyon Kaynağı": label,
                "SS": round(ss_val, 4),
                "df": int(row["df"]),
                "F": round(float(row["F"]), 4) if not pd.isna(row["F"]) else "-",
                "p": fmt_p(row["PR(>F)"]) if not pd.isna(row["PR(>F)"]) else "-",
                "η²": eta_sq if is_factor else "-",
                "ω²": omega_sq if is_factor else "-"
            })
        tables.append({"title": f"Tek Yönlü ANOVA Tablosu ({var})", "rows": anova_rows})

        groups_samples = [g[var].values for _, g in sub_df.groupby(req.group_var)]
        lev_stat, lev_p = stats.levene(*groups_samples, center="median")
        equal_var = lev_p >= req.alpha

        if equal_var:
            tukey = pairwise_tukeyhsd(endog=sub_df[var], groups=sub_df[req.group_var].astype(str), alpha=req.alpha)
            tukey_summary = tukey.summary().data
            posthoc_rows = []
            non_sig_pairs = []
            for r in tukey_summary[1:]:
                g1, g2, reject = str(r[0]), str(r[1]), bool(r[6])
                if not reject:
                    non_sig_pairs.append((g1, g2))
                posthoc_rows.append({
                    "Grup 1": g1,
                    "Grup 2": g2,
                    "Ort. Farkı": round(float(r[2]), 4),
                    "p-adj": fmt_p(float(r[3])),
                    "Alt (%95)": round(float(r[4]), 4),
                    "Üst (%95)": round(float(r[5]), 4),
                    "Anlamlı": "Evet *" if reject else "Hayır"
                })
            posthoc_title = "Post-Hoc Çoklu Karşılaştırma (Tukey HSD — Varyanslar Homojen)"
        else:
            posthoc_rows, non_sig_pairs = games_howell_pairs(sub_df, var, req.group_var, req.alpha)
            posthoc_title = "Post-Hoc Çoklu Karşılaştırma (Games-Howell — Varyans Homojenliği İhlal Edildi)"
            notes.append(f"Levene testi anlamlı çıktı (F = {lev_stat:.3f}, p {fmt_p(lev_p)}); varyanslar eşit olmadığı için otomatik Games-Howell post-hoc uygulandı.")

        tables.append({"title": posthoc_title, "rows": posthoc_rows})

        grp_stats = sub_df.groupby(req.group_var)[var].agg(["count", "mean", "std", "sem"]).reset_index()
        means_dict = {str(r[req.group_var]): float(r["mean"]) for _, r in grp_stats.iterrows()}
        letters_map = compute_compact_letters_from_pairs(means_dict, non_sig_pairs)

        letter_rows = []
        for _, r in grp_stats.sort_values("mean", ascending=False).iterrows():
            g_name = str(r[req.group_var])
            letter_rows.append({
                "Grup / Uygulama": g_name,
                "N": int(r["count"]),
                "Ortalama ± SE": f"{r['mean']:.3f} ± {r['sem']:.3f}",
                "Std. Sapma": round(float(r["std"]), 3),
                "Grup Harfi (p<0.05)": letters_map.get(g_name, "a")
            })
        tables.append({"title": "Grup Ortalamaları ve İstatistiksel Harflendirme", "rows": letter_rows})

        apa_note = f"Tek yönlü varyans analizi (ANOVA) sonuçları, {req.group_var} faktörünün {var} üzerinde F({df1_main}, {df2_main}) = {f_val_main}, p {fmt_p(p_val_main)}, η² = {eta_sq} düzeyinde etkili olduğunu göstermiştir."
        chart_data = {
            "type": "bar_error",
            "title": f"{req.group_var} Gruplarına Göre {var} Ortalamaları (±SE)",
            "labels": [f"{r[req.group_var]} ({letters_map.get(str(r[req.group_var]), '')})" for _, r in grp_stats.iterrows()],
            "values": [round(float(x), 3) for x in grp_stats["mean"]],
            "errors": [round(float(x), 3) for x in grp_stats["sem"]],
            "y_label": var
        }
        r_code = f"model <- aov({var} ~ as.factor({req.group_var}), data = df)\nsummary(model)\nTukeyHSD(model)"

    # 8. İKİ YÖNLÜ ANOVA + HÜCRE HARFLENDİRMESİ
    elif req.test_type == "factorial_anova":
        report_title = "İki Yönlü ANOVA (A×B)"
        if not req.group_var or not req.second_factor or not req.dependent_vars:
            raise HTTPException(status_code=400, detail="Bağımlı değişken, Faktör 1 ve Faktör 2 kutularını doldurun.")

        var = req.dependent_vars[0]
        f1, f2 = req.group_var, req.second_factor
        sub_df = df[[var, f1, f2]].dropna()
        sub_df[var] = pd.to_numeric(sub_df[var], errors="coerce")
        sub_df = sub_df.dropna()

        formula = f'Q("{var}") ~ C(Q("{f1}")) + C(Q("{f2}")) + C(Q("{f1}")):C(Q("{f2}"))'
        model = ols(formula, data=sub_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        sse = float(anova_table.loc["Residual", "sum_sq"])

        anova_rows = []
        for idx, row in anova_table.iterrows():
            s_idx = str(idx)
            if "Intercept" in s_idx:
                continue
            if ":" in s_idx:
                label = f"{f1} × {f2} (İnteraksiyon)"
            elif f1 in s_idx:
                label = f1
            elif f2 in s_idx:
                label = f2
            else:
                label = "Hata (Residual)"
            ss_v = float(row["sum_sq"])
            p_eta = round(ss_v / (ss_v + sse), 3) if label != "Hata (Residual)" and (ss_v + sse) > 0 else "-"
            anova_rows.append({
                "Varyasyon Kaynağı": label,
                "SS (Tip III)": round(ss_v, 4),
                "df": int(row["df"]),
                "F": round(float(row["F"]), 4) if not pd.isna(row["F"]) else "-",
                "p": fmt_p(row["PR(>F)"]) if not pd.isna(row["PR(>F)"]) else "-",
                "Kısmi η²": p_eta
            })
        tables.append({"title": f"İki Yönlü ANOVA Tablosu ({var})", "rows": anova_rows})

        sub_df["Hucre"] = sub_df[f1].astype(str) + " × " + sub_df[f2].astype(str)
        tukey = pairwise_tukeyhsd(endog=sub_df[var], groups=sub_df["Hucre"], alpha=req.alpha)
        non_sig_pairs = [(str(r[0]), str(r[1])) for r in tukey.summary().data[1:] if not bool(r[6])]
        cell_stats = sub_df.groupby("Hucre")[var].agg(["count", "mean", "std", "sem"]).reset_index()
        means_dict = {str(r["Hucre"]): float(r["mean"]) for _, r in cell_stats.iterrows()}
        letters_map = compute_compact_letters_from_pairs(means_dict, non_sig_pairs)

        cell_rows = []
        for _, r in cell_stats.sort_values("mean", ascending=False).iterrows():
            c_name = str(r["Hucre"])
            cell_rows.append({
                "Faktör 1 × Faktör 2": c_name,
                "N": int(r["count"]),
                "Ortalama ± SE": f"{r['mean']:.3f} ± {r['sem']:.3f}",
                "Harf": letters_map.get(c_name, "a")
            })
        tables.append({"title": "Hücre Ortalamaları ve İnteraksiyon Harflendirmesi", "rows": cell_rows})
        apa_note = f"{f1} ve {f2} faktörlerinin {var} üzerindeki ana etkileri ve {f1} × {f2} etkileşimi Tip III iki yönlü ANOVA ile değerlendirilmiştir."
        r_code = f"model <- aov({var} ~ as.factor({f1}) * as.factor({f2}), data = df)\nsummary(model)\nTukeyHSD(model)"

    # 9. TEKRARLI ÖLÇÜMLER ANOVA + GREENHOUSE-GEISSER
    elif req.test_type == "rm_anova":
        report_title = "Tekrarlı Ölçümler ANOVA"
        if len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="En az 2 tekrarlı ölçüm sütunu seçin.")

        sub = df[req.dependent_vars].apply(pd.to_numeric, errors="coerce").dropna().reset_index()
        sub = sub.rename(columns={"index": "Denek_ID"})
        long_df = sub.melt(id_vars=["Denek_ID"], value_vars=req.dependent_vars, var_name="Zaman_Faktoru", value_name="Olcum")

        aovrm = AnovaRM(long_df, depvar="Olcum", subject="Denek_ID", within=["Zaman_Faktoru"]).fit()
        rm_tbl = aovrm.anova_table.reset_index()
        row0 = rm_tbl.iloc[0]
        f_val = float(row0["F Value"])
        df1 = float(row0["Num DF"])
        df2 = float(row0["Den DF"])
        p_unc = float(row0["Pr > F"])

        # Greenhouse-Geisser epsilon hesabı
        Y = sub[req.dependent_vars].values
        Y_centered = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + Y.mean()
        S = np.cov(Y_centered, rowvar=False)
        k = Y.shape[1]
        trace_s = np.trace(S)
        trace_s2 = np.sum(S ** 2)
        gg_eps = (trace_s ** 2) / ((k - 1) * trace_s2) if (k > 1 and trace_s2 > 0) else 1.0
        gg_eps = min(1.0, max(1.0 / max(1, k - 1), gg_eps))
        p_gg = float(stats.f.sf(f_val, df1 * gg_eps, df2 * gg_eps))

        tables.append({
            "title": "Tekrarlı Ölçümler ANOVA & Küresellik Düzeltmesi",
            "rows": [
                {"Düzeltme": "Küresellik Varsayımı (Düzeltmesiz)", "F": round(f_val, 4), "df1": round(df1, 1), "df2": round(df2, 1), "p": fmt_p(p_unc)},
                {"Düzeltme": f"Greenhouse-Geisser (ε = {gg_eps:.3f})", "F": round(f_val, 4), "df1": round(df1 * gg_eps, 2), "df2": round(df2 * gg_eps, 2), "p": fmt_p(p_gg)}
            ]
        })
        apa_note = f"Tekrarlı ölçümler ANOVA analizinde zaman faktörünün etkisi F({df1:.0f}, {df2:.0f}) = {f_val:.3f}, p {fmt_p(p_unc)} (Greenhouse-Geisser düzeltmeli p {fmt_p(p_gg)}) olarak bulunmuştur."
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"sub <- na.omit(df[, c({vars_str})])\nlong <- stack(sub)\nsummary(aov(values ~ ind, data = long))"

    # 10. ANCOVA
    elif req.test_type == "ancova":
        report_title = "ANCOVA Kovaryans Analizi"
        if not req.group_var or not req.second_factor or not req.dependent_vars:
            raise HTTPException(status_code=400, detail="Bağımlı değişken, Grup Faktörü ve Sayısal Kovaryant kutularını doldurun.")

        var, grp, cov = req.dependent_vars[0], req.group_var, req.second_factor
        sub_df = df[[var, grp, cov]].dropna()
        sub_df[var] = pd.to_numeric(sub_df[var], errors="coerce")
        sub_df[cov] = pd.to_numeric(sub_df[cov], errors="coerce")
        sub_df = sub_df.dropna()

        formula = f'Q("{var}") ~ C(Q("{grp}")) + Q("{cov}")'
        model = ols(formula, data=sub_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        sse = float(anova_table.loc["Residual", "sum_sq"])

        ancova_rows = []
        for idx, row in anova_table.iterrows():
            s_idx = str(idx)
            if "Intercept" in s_idx:
                continue
            if "C(" in s_idx:
                label = f"{grp} (Düzeltilmiş Grup Etkisi)"
            elif cov in s_idx:
                label = f"{cov} (Kovaryant)"
            else:
                label = "Hata (Residual)"
            ss_v = float(row["sum_sq"])
            ancova_rows.append({
                "Varyasyon Kaynağı": label,
                "SS (Tip III)": round(ss_v, 4),
                "df": int(row["df"]),
                "F": round(float(row["F"]), 4) if not pd.isna(row["F"]) else "-",
                "p": fmt_p(row["PR(>F)"]) if not pd.isna(row["PR(>F)"]) else "-",
                "Kısmi η²": round(ss_v / (ss_v + sse), 3) if label != "Hata (Residual)" else "-"
            })
        tables.append({"title": f"ANCOVA Kovaryans Analizi ({var} | Kovaryant: {cov})", "rows": ancova_rows})
        apa_note = f"{cov} kovaryantı kontrol altına alındığında {grp} gruplarının {var} üzerindeki düzeltilmiş etkisi ANCOVA ile incelenmiştir."
        r_code = f"fit <- aov({var} ~ {cov} + as.factor({grp}), data = df)\nsummary(fit)"

    # 11. MANOVA (YAPISAL VERİ OKUMA)
    elif req.test_type == "manova":
        report_title = "MANOVA Çok Değişkenli Analiz"
        if not req.group_var or len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="MANOVA için en az 2 Bağımlı Değişken ve 1 Grup Faktörü seçin.")

        sub_df = df[req.dependent_vars + [req.group_var]].dropna()
        for v in req.dependent_vars:
            sub_df[v] = pd.to_numeric(sub_df[v], errors="coerce")
        sub_df = sub_df.dropna()

        lhs = " + ".join([f'Q("{v}")' for v in req.dependent_vars])
        maov = MANOVA.from_formula(f'{lhs} ~ C(Q("{req.group_var}"))', data=sub_df)
        mv_res = maov.mv_test()
        term_key = [k for k in mv_res.results.keys() if "Intercept" not in k][0]
        stat_df = mv_res.results[term_key]["stat"]

        mv_rows = []
        for metric_name, r in stat_df.iterrows():
            mv_rows.append({
                "Çok Değişkenli Test": str(metric_name),
                "Değer": round(float(r["Value"]), 4),
                "Num DF": int(r["Num DF"]),
                "Den DF": int(r["Den DF"]),
                "F": round(float(r["F Value"]), 4),
                "Pr > F": fmt_p(float(r["Pr > F"]))
            })
        tables.append({"title": f"MANOVA Çok Değişkenli Test Sonuçları ({req.group_var})", "rows": mv_rows})
        w0 = mv_rows[0]
        apa_note = f"MANOVA sonucunda {req.group_var} grupları arasında bağımlı değişkenlerin birleşik vektörü Wilks' Λ = {w0['Değer']}, F({w0['Num DF']}, {w0['Den DF']}) = {w0['F']}, p {w0['Pr > F']} düzeyinde değerlendirilmiştir."
        r_code = f"fit <- manova(cbind({', '.join(req.dependent_vars)}) ~ as.factor({req.group_var}), data = df)\nsummary(fit, test = 'Wilks')"

    # 12. PARAMETRİK OLMAYAN TESTLER (MANN-WHITNEY / KRUSKAL)
    elif req.test_type == "nonparam":
        report_title = "Parametrik Olmayan Grup Testi"
        if not req.group_var or not req.dependent_vars:
            raise HTTPException(status_code=400, detail="Bağımlı değişken ve gruplama değişkeni seçin.")
        groups = df[req.group_var].dropna().unique()
        rows = []
        for var in req.dependent_vars:
            samples = [pd.to_numeric(df[df[req.group_var] == g][var], errors="coerce").dropna() for g in groups]
            if len(groups) == 2:
                stat, p_val = stats.mannwhitneyu(samples[0], samples[1])
                test_name = "Mann-Whitney U"
            else:
                stat, p_val = stats.kruskal(*samples)
                test_name = "Kruskal-Wallis H"
            rows.append({
                "Değişken": var,
                "Test": test_name,
                "Grup Sayısı": len(groups),
                "İstatistik": round(float(stat), 4),
                "p": fmt_p(p_val)
            })
        tables.append({"title": "Parametrik Olmayan Grup Karşılaştırması", "rows": rows})
        apa_note = f"{rows[0]['Test']} analizi sonucunda {req.group_var} grupları arasında {rows[0]['Değişken']} dağılımı açısından p {rows[0]['p']} elde edilmiştir."
        r_code = f"kruskal.test({req.dependent_vars[0]} ~ as.factor({req.group_var}), data = df)"

    # 13. KORELASYON ANALİZİ + ISI HARİTASI
    elif req.test_type == "correlation":
        report_title = "Korelasyon Matrisi & Heatmap"
        if len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="En az 2 sayısal değişken seçin.")
        num_df = df[req.dependent_vars].apply(pd.to_numeric, errors="coerce").dropna()
        corr_rows = []
        for i in range(len(req.dependent_vars)):
            for j in range(i + 1, len(req.dependent_vars)):
                v1, v2 = req.dependent_vars[i], req.dependent_vars[j]
                r_val, p_val = stats.pearsonr(num_df[v1], num_df[v2])
                rho_val, sp_p = stats.spearmanr(num_df[v1], num_df[v2])
                corr_rows.append({
                    "Değişken 1": v1,
                    "Değişken 2": v2,
                    "Pearson r": round(float(r_val), 4),
                    "Pearson p": fmt_p(p_val),
                    "Spearman rho": round(float(rho_val), 4),
                    "Spearman p": fmt_p(sp_p)
                })
        tables.append({"title": "Korelasyon Matrisi (Pearson & Spearman)", "rows": corr_rows})
        r0 = corr_rows[0]
        apa_note = f"{r0['Değişken 1']} ile {r0['Değişken 2']} arasında Pearson korelasyon katsayısı r = {r0['Pearson r']} (p {r0['Pearson p']}) olarak hesaplanmıştır."

        corr_mat = num_df.corr().round(3)
        chart_data = {
            "type": "heatmap",
            "title": "Pearson Korelasyon Isı Haritası (Heatmap)",
            "labels": req.dependent_vars,
            "matrix": corr_mat.values.tolist()
        }
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"cor(df[, c({vars_str})], use = 'complete.obs')"

    # 14. ÇOKLU DOĞRUSAL REGRESYON + VIF + DURBIN-WATSON
    elif req.test_type == "regression":
        report_title = "Çoklu Doğrusal Regresyon"
        if len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="İlk seçilen değişken Bağımlı (Y), diğerleri Bağımsız (X) alınır.")
        y_var = req.dependent_vars[0]
        x_vars = req.dependent_vars[1:]
        sub = df[[y_var] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
        X = sm.add_constant(sub[x_vars])
        model = sm.OLS(sub[y_var], X).fit()
        dw_stat = float(durbin_watson(model.resid))

        tables.append({
            "title": "Regresyon Model Özeti & Tanı İstatistikleri",
            "rows": [{
                "Bağımlı (Y)": y_var,
                "N": int(model.nobs),
                "R²": round(float(model.rsquared), 4),
                "Düzeltilmiş R²": round(float(model.rsquared_adj), 4),
                "F": round(float(model.fvalue), 4),
                "Model p": fmt_p(float(model.f_pvalue)),
                "Durbin-Watson": round(dw_stat, 3)
            }]
        })

        coef_rows = []
        for idx_p, param in enumerate(model.params.index):
            vif_val = "-"
            if param != "const" and len(x_vars) >= 2:
                vif_val = round(float(variance_inflation_factor(X.values, idx_p)), 2)
            coef_rows.append({
                "Terim": "Sabit (Intercept)" if param == "const" else param,
                "Katsayı (B)": round(float(model.params[param]), 4),
                "Std. Hata": round(float(model.bse[param]), 4),
                "t": round(float(model.tvalues[param]), 4),
                "p": fmt_p(float(model.pvalues[param])),
                "VIF": vif_val
            })
        tables.append({"title": "Regresyon Katsayıları ve VIF Çoklu Bağıntı Kontrolü", "rows": coef_rows})
        apa_note = f"Kurulan çoklu doğrusal regresyon modeli {y_var} değişkenindeki toplam varyansın %{model.rsquared*100:.1f}'ini (Düzeltilmiş R² = {model.rsquared_adj:.3f}) açıklamaktadır, F = {model.fvalue:.3f}, p {fmt_p(float(model.f_pvalue))}."
        r_code = f"model <- lm({y_var} ~ {' + '.join(x_vars)}, data = df)\nsummary(model)"

    # 15. Kİ-KARE / FISHER KESİN TESTİ
    elif req.test_type == "chisq":
        report_title = "Ki-Kare & Fisher Testi"
        if not req.group_var or not req.dependent_vars:
            raise HTTPException(status_code=400, detail="İki kategorik değişken seçin.")
        v1, v2 = req.dependent_vars[0], req.group_var
        ct = pd.crosstab(df[v1], df[v2])
        chi2, p_val, dof, expected = stats.chi2_contingency(ct, correction=False)
        n_tot = ct.values.sum()
        cramer_v = math.sqrt(chi2 / (n_tot * max(1, min(ct.shape) - 1)))

        test_rows = [{
            "Test": "Pearson Ki-Kare (χ²)",
            "İstatistik": round(float(chi2), 4),
            "df": int(dof),
            "p": fmt_p(p_val),
            "Cramér's V": round(cramer_v, 3)
        }]
        if ct.shape == (2, 2):
            odds_r, fisher_p = stats.fisher_exact(ct.values)
            test_rows.append({
                "Test": "Fisher Kesin Testi (2×2)",
                "İstatistik": f"OR = {odds_r:.3f}",
                "df": "-",
                "p": fmt_p(float(fisher_p)),
                "Cramér's V": round(cramer_v, 3)
            })
        if np.min(expected) < 5:
            notes.append("Beklenen frekansı 5'ten küçük hücreler mevcut; Fisher Kesin Testi sonucunu dikkate alın.")

        tables.append({"title": f"Ki-Kare Bağımsızlık Testi ({v1} × {v2})", "rows": test_rows})
        tables.append({"title": "Çapraz Tablo (Kontenjans)", "rows": ct.reset_index().to_dict(orient="records")})
        apa_note = f"{v1} ve {v2} kategorik değişkenleri arasındaki ilişki Ki-kare bağımsızlık testi ile incelenmiştir, χ²({dof}) = {chi2:.3f}, p {fmt_p(p_val)}, Cramér's V = {cramer_v:.3f}."
        r_code = f"chisq.test(table(df[['{v1}']], df[['{v2}']]))"

    # 16. MCNEMAR TESTİ
    elif req.test_type == "mcnemar":
        report_title = "McNemar Eşleşmiş Kategorik Testi"
        if len(req.dependent_vars) != 2:
            raise HTTPException(status_code=400, detail="McNemar testi için 2 ikili (0/1) ölçüm değişkeni seçin.")
        v1, v2 = req.dependent_vars[0], req.dependent_vars[1]
        ct = pd.crosstab(df[v1], df[v2])
        if ct.shape != (2, 2):
            raise HTTPException(status_code=400, detail="McNemar testi için iki değişken de tam 2 kategorili olmalıdır.")
        b = int(ct.iloc[0, 1])
        c = int(ct.iloc[1, 0])
        chi2_mc = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
        p_mc = float(stats.chi2.sf(chi2_mc, 1)) if (b + c) > 0 else 1.0

        tables.append({
            "title": f"McNemar Testi ({v1} → {v2})",
            "rows": [{"McNemar χ²": round(chi2_mc, 4), "df": 1, "p": fmt_p(p_mc), "Değişen Gözlem (b+c)": b + c}]
        })
        apa_note = f"McNemar testi sonucunda {v1} ve {v2} ölçümleri arasındaki kategorik değişim χ²(1) = {chi2_mc:.3f}, p {fmt_p(p_mc)} olarak bulunmuştur."
        r_code = f"mcnemar.test(table(df[['{v1}']], df[['{v2}']]))"

    # 17. CRONBACH'S ALPHA + MADDE ANALİZİ
    elif req.test_type == "cronbach":
        report_title = "Cronbach's Alpha Güvenilirlik"
        if len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="Ölçek güvenilirliği için en az 2 madde seçin.")
        sub = df[req.dependent_vars].apply(pd.to_numeric, errors="coerce").dropna()
        k = sub.shape[1]
        item_vars = sub.var(axis=0, ddof=1)
        total_var = sub.sum(axis=1).var(ddof=1)
        alpha_val = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var != 0 else 0.0

        tables.append({
            "title": "Ölçek Güvenilirlik Özeti (Cronbach's Alpha)",
            "rows": [{
                "Madde Sayısı (k)": k,
                "Örneklem (N)": len(sub),
                "Cronbach's Alpha (α)": round(float(alpha_val), 4),
                "Değerlendirme": "Yüksek Güvenilirlik" if alpha_val >= 0.8 else ("Kabul Edilebilir" if alpha_val >= 0.7 else "Düşük Güvenilirlik")
            }]
        })

        item_rows = []
        for col in req.dependent_vars:
            rem = sub.drop(columns=[col])
            k_r = rem.shape[1]
            v_r = rem.var(axis=0, ddof=1).sum()
            t_r = rem.sum(axis=1).var(ddof=1)
            alpha_drop = (k_r / (k_r - 1)) * (1 - v_r / t_r) if (k_r > 1 and t_r != 0) else 0.0
            r_tot, _ = stats.pearsonr(sub[col], rem.sum(axis=1))
            item_rows.append({
                "Madde": col,
                "Ortalama": round(float(sub[col].mean()), 3),
                "SD": round(float(sub[col].std(ddof=1)), 3),
                "Düzeltilmiş Madde-Toplam r": round(float(r_tot), 3),
                "Madde Silinirse α": round(float(alpha_drop), 4)
            })
        tables.append({"title": "Madde-Toplam İstatistikleri", "rows": item_rows})
        apa_note = f"{k} maddeden oluşan ölçeğin iç tutarlılık katsayısı Cronbach's α = {alpha_val:.3f} olarak bulunmuştur."
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"items <- na.omit(df[, c({vars_str})])\nk <- ncol(items)\n(k / (k - 1)) * (1 - sum(apply(items, 2, var)) / var(rowSums(items)))"

    # 18. EFA + KMO + BARTLETT KÜRESELLİK TESTİ
    elif req.test_type == "efa":
        report_title = "Açımlayıcı Faktör Analizi (EFA)"
        if len(req.dependent_vars) < 3:
            raise HTTPException(status_code=400, detail="Açımlayıcı Faktör Analizi (EFA) için en az 3 ölçek maddesi seçin.")

        sub = df[req.dependent_vars].apply(pd.to_numeric, errors="coerce").dropna()
        n_obs, p_vars = sub.shape
        corr_mat = np.corrcoef(sub.values, rowvar=False)
        inv_corr = np.linalg.pinv(corr_mat)
        d_mat = np.diag(1.0 / np.sqrt(np.diag(inv_corr)))
        part_corr = -d_mat @ inv_corr @ d_mat
        np.fill_diagonal(part_corr, 0.0)
        corr_sq = corr_mat ** 2
        np.fill_diagonal(corr_sq, 0.0)
        kmo_val = float(np.sum(corr_sq) / (np.sum(corr_sq) + np.sum(part_corr ** 2)))

        det_r = max(1e-12, float(np.linalg.det(corr_mat)))
        bart_chi2 = -(n_obs - 1 - (2 * p_vars + 5) / 6.0) * math.log(det_r)
        bart_df = int(p_vars * (p_vars - 1) / 2)
        bart_p = float(stats.chi2.sf(bart_chi2, bart_df))

        tables.append({
            "title": "KMO Örneklem Yeterliliği ve Bartlett Küresellik Testi",
            "rows": [{
                "Kaiser-Meyer-Olkin (KMO)": round(kmo_val, 3),
                "Bartlett Ki-Kare (χ²)": round(bart_chi2, 3),
                "df": bart_df,
                "p": fmt_p(bart_p),
                "Uygunluk": "Faktör Analizine Uygun" if (kmo_val >= 0.6 and bart_p < 0.05) else "Sınırda / Zayıf"
            }]
        })

        scaled = StandardScaler().fit_transform(sub)
        n_factors = min(2, p_vars - 1)
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(scaled)

        load_rows = []
        for idx, var_name in enumerate(req.dependent_vars):
            row_dict = {"Madde / Değişken": var_name}
            comm = 0.0
            for f_i in range(n_factors):
                val = float(fa.components_[f_i, idx])
                comm += val ** 2
                row_dict[f"Faktör {f_i+1} Yükü"] = round(val, 4)
            row_dict["Ortak Varyans (h²)"] = round(comm, 4)
            load_rows.append(row_dict)

        tables.append({"title": f"Faktör Yükleri ({n_factors} Faktörlü Çözüm)", "rows": load_rows})
        apa_note = f"Açımlayıcı Faktör Analizi öncesi KMO = {kmo_val:.3f} ve Bartlett küresellik testi χ²({bart_df}) = {bart_chi2:.2f}, p {fmt_p(bart_p)} olarak hesaplanmıştır."
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"factanal(na.omit(df[, c({vars_str})]), factors = 1)"

    # 19. PCA
    elif req.test_type == "pca":
        report_title = "Temel Bileşenler Analizi (PCA)"
        if len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="PCA için en az 2 sayısal değişken seçin.")
        sub = df[req.dependent_vars].apply(pd.to_numeric, errors="coerce").dropna()
        scaled = StandardScaler().fit_transform(sub)
        pca = PCA()
        scores = pca.fit_transform(scaled)

        var_rows = []
        cum_var = 0.0
        for i, (ev, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
            cum_var += ratio * 100
            var_rows.append({
                "Bileşen": f"PC{i+1}",
                "Özdeğer (Eigenvalue)": round(float(ev), 4),
                "Açıklanan Varyans (%)": round(float(ratio * 100), 2),
                "Kümülatif Varyans (%)": round(float(cum_var), 2)
            })
        tables.append({"title": "PCA Özdeğerler ve Açıklanan Varyans", "rows": var_rows})
        apa_note = f"Temel Bileşenler Analizi (PCA) sonucunda ilk iki bileşen (PC1 ve PC2) toplam varyansın %{(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])*100:.1f}'ini açıklamaktadır."

        chart_data = {
            "type": "scatter",
            "title": "PCA Skor Dağılımı (PC1 vs PC2)",
            "points": [{"x": round(float(scores[i, 0]), 3), "y": round(float(scores[i, 1]), 3)} for i in range(len(scores))],
            "x_label": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            "y_label": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
        }
        vars_str = ", ".join([f'"{v}"' for v in req.dependent_vars])
        r_code = f"pca_res <- prcomp(na.omit(df[, c({vars_str})]), scale. = TRUE)\nsummary(pca_res)"

    # 20. LOJİSTİK REGRESYON & ROC
    elif req.test_type == "roc_logistic":
        report_title = "İkili Lojistik Regresyon & ROC"
        if len(req.dependent_vars) < 2:
            raise HTTPException(status_code=400, detail="İlk seçilen değişken 0/1 Sonuç (Y), diğerleri Tahmin Edici (X) olmalıdır.")
        y_var = req.dependent_vars[0]
        x_vars = req.dependent_vars[1:]
        sub = df[[y_var] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()

        if set(sub[y_var].unique()) - {0, 1}:
            raise HTTPException(status_code=400, detail=f"Bağımlı değişken ({y_var}) yalnızca 0 ve 1 değerlerinden oluşmalıdır.")

        X = sm.add_constant(sub[x_vars])
        logit_model = sm.Logit(sub[y_var], X).fit(disp=False)
        probs = logit_model.predict(X)
        fpr, tpr, _ = roc_curve(sub[y_var], probs)
        roc_auc = auc(fpr, tpr)

        coef_rows = []
        for param in logit_model.params.index:
            b_val = float(logit_model.params[param])
            coef_rows.append({
                "Terim": "Sabit" if param == "const" else param,
                "Katsayı (B)": round(b_val, 4),
                "Odds Ratio (Exp B)": round(float(np.exp(b_val)), 4),
                "z": round(float(logit_model.tvalues[param]), 4),
                "p": fmt_p(float(logit_model.pvalues[param]))
            })
        tables.append({"title": f"İkili Lojistik Regresyon & Odds Ratio (AUC = {roc_auc:.3f})", "rows": coef_rows})
        apa_note = f"İkili lojistik regresyon modelinin ROC eğrisi altında kalan alanı AUC = {roc_auc:.3f} olarak hesaplanmıştır."

        chart_data = {
            "type": "line",
            "title": f"ROC Eğrisi (AUC = {roc_auc:.3f})",
            "points": [{"x": round(float(fx), 3), "y": round(float(ty), 3)} for fx, ty in zip(fpr, tpr)],
            "x_label": "1 - Özgüllük (FPR)",
            "y_label": "Duyarlılık (TPR)"
        }
        r_code = f"fit <- glm({y_var} ~ {' + '.join(x_vars)}, data = df, family = binomial)\nsummary(fit)"

    # 21. SAĞKALIM ANALİZİ (KAPLAN-MEIER)
    elif req.test_type == "survival":
        report_title = "Kaplan-Meier Sağkalım Analizi"
        if len(req.dependent_vars) != 2:
            raise HTTPException(status_code=400, detail="Sağkalım analizi için ilk olarak Süre (Zaman), ikinci olarak Olay Durumu (0/1) değişkenlerini seçin.")

        t_col, e_col = req.dependent_vars[0], req.dependent_vars[1]
        sub = df[[t_col, e_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(t_col)
        times = sub[t_col].values
        events = sub[e_col].values

        unique_times = np.unique(times[events == 1])
        surv_prob = 1.0
        km_rows = []
        km_points = [{"x": 0.0, "y": 1.0}]

        for t in unique_times:
            at_risk = np.sum(times >= t)
            d_i = np.sum((times == t) & (events == 1))
            surv_prob *= (1.0 - d_i / at_risk)
            km_rows.append({
                "Zaman": round(float(t), 2),
                "Risk Altındaki (N)": int(at_risk),
                "Olay Sayısı": int(d_i),
                "Sağkalım Olasılığı": round(float(surv_prob), 4)
            })
            km_points.append({"x": round(float(t), 2), "y": round(float(surv_prob), 4)})

        tables.append({"title": f"Kaplan-Meier Sağkalım Tablosu ({t_col})", "rows": km_rows[:25]})
        apa_note = f"Kaplan-Meier sağkalım analizinde takip süresi sonunda kümülatif sağkalım olasılığı {surv_prob:.3f} olarak tahmin edilmiştir."
        chart_data = {
            "type": "line",
            "title": "Kaplan-Meier Sağkalım Eğrisi",
            "points": km_points,
            "x_label": t_col,
            "y_label": "Kümülatif Sağkalım Olasılığı"
        }
        r_code = f"sub <- na.omit(df[, c('{t_col}', '{e_col}')])\nsummary(sub)"

    # 22. COX ORANTILI HAZARD REGRESYONU
    elif req.test_type == "cox":
        report_title = "Cox Orantılı Hazard Regresyonu"
        if len(req.dependent_vars) < 3:
            raise HTTPException(status_code=400, detail="Cox regresyonu için sırayla: 1. Süre, 2. Olay (0/1), 3+. Açıklayıcı Değişken(ler) seçin.")
        t_col, e_col = req.dependent_vars[0], req.dependent_vars[1]
        cov_cols = req.dependent_vars[2:]
        sub = df[[t_col, e_col] + cov_cols].apply(pd.to_numeric, errors="coerce").dropna()

        mod = PHReg(sub[t_col].values, sub[cov_cols].values, status=sub[e_col].values)
        res_cox = mod.fit()
        cox_rows = []
        for idx_c, cname in enumerate(cov_cols):
            b_v = float(res_cox.params[idx_c])
            se_v = float(res_cox.bse[idx_c])
            z_v = b_v / se_v if se_v > 0 else 0.0
            p_v = float(2 * stats.norm.sf(abs(z_v)))
            cox_rows.append({
                "Değişken": cname,
                "Katsayı (B)": round(b_v, 4),
                "Std. Hata": round(se_v, 4),
                "Hazard Ratio (HR)": round(float(np.exp(b_v)), 4),
                "z": round(z_v, 4),
                "p": fmt_p(p_v)
            })
        tables.append({"title": f"Cox Orantılı Hazard Modeli (Süre: {t_col}, Olay: {e_col})", "rows": cox_rows})
        c0 = cox_rows[0]
        apa_note = f"Cox orantılı hazard regresyonunda {c0['Değişken']} değişkeni için Hazard Ratio HR = {c0['Hazard Ratio (HR)']}, p {c0['p']} olarak hesaplanmıştır."
        r_code = f"sub <- na.omit(df[, c('{t_col}', '{e_col}', {', '.join([repr(c) for c in cov_cols])})])\nsummary(sub)"

    # 23. ZAMAN SERİSİ + MANN-KENDALL + DURBIN-WATSON
    elif req.test_type == "timeseries":
        report_title = "Zaman Serisi & Trend Analizi"
        if not req.dependent_vars:
            raise HTTPException(status_code=400, detail="Zaman serisi analizi için en az 1 sayısal seri sütunu seçin.")

        var = req.dependent_vars[0]
        series = pd.to_numeric(df[var], errors="coerce").dropna().reset_index(drop=True)
        t_idx = np.arange(1, len(series) + 1)
        slope, intercept, r_val, p_val, _ = stats.linregress(t_idx, series.values)
        mk_tau, mk_p = stats.kendalltau(t_idx, series.values)
        resid = series.values - (intercept + slope * t_idx)
        dw_ts = float(durbin_watson(resid))
        ma_3 = series.rolling(window=3, min_periods=1).mean()

        tables.append({
            "title": f"Zaman Serisi Trend ve Otokorelasyon Analizi ({var})",
            "rows": [{
                "Gözlem (T)": len(series),
                "Eğim (β1)": round(float(slope), 4),
                "R²": round(float(r_val ** 2), 4),
                "Trend p": fmt_p(float(p_val)),
                "Mann-Kendall τ": round(float(mk_tau), 3),
                "MK p": fmt_p(float(mk_p)),
                "Durbin-Watson": round(dw_ts, 3)
            }]
        })
        apa_note = f"{var} zaman serisinde dönem başına ortalama değişim eğimi β1 = {slope:.4f} (R² = {r_val**2:.3f}, p {fmt_p(float(p_val))}), Mann-Kendall τ = {mk_tau:.3f} (p {fmt_p(float(mk_p))}) bulunmuştur."
        chart_data = {
            "type": "line",
            "title": f"{var} — 3 Dönemlik Hareketli Ortalama Eğrisi",
            "points": [{"x": int(i + 1), "y": round(float(v), 3)} for i, v in enumerate(ma_3)],
            "x_label": "Zaman / Gözlem Sırası (t)",
            "y_label": f"{var} (MA-3)"
        }
        r_code = f"ts_data <- ts(na.omit(df[['{var}']]))\nsummary(lm(ts_data ~ time(ts_data)))\ncor.test(seq_along(ts_data), as.numeric(ts_data), method = 'kendall')"

    REPORT_HISTORY[req.session_id].append({"title": report_title, "tables": tables, "r_code": r_code, "apa_note": apa_note, "notes": notes})
    return {"tables": tables, "r_code": r_code, "chart": chart_data, "apa_note": apa_note, "notes": notes}


@app.get("/api/export-docx")
async def export_apa_word(session_id: str = "default"):
    history = REPORT_HISTORY.get(session_id, [])
    if not history:
        raise HTTPException(status_code=404, detail="Dışa aktarılacak aktif analiz sonucu bulunamadı.")

    doc = Document()
    doc.add_heading("Akademik İstatistik & Veri Analizi Raporu (APA)", level=1)

    for idx_rep, item in enumerate(history, start=1):
        doc.add_heading(f"{idx_rep}. {item['title']}", level=2)
        if item.get("apa_note"):
            p_note = doc.add_paragraph(item["apa_note"])
            p_note.runs[0].italic = True

        for tbl in item.get("tables", []):
            rows = tbl.get("rows", [])
            if not rows:
                continue
            p_tbl = doc.add_paragraph(tbl["title"])
            p_tbl.runs[0].bold = True

            headers = list(rows[0].keys())
            word_table = doc.add_table(rows=1, cols=len(headers))
            word_table.style = "Table Grid"

            for i, h in enumerate(headers):
                cell = word_table.rows[0].cells[i]
                cell.text = str(h)
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.bold = True
                        run.font.size = Pt(9.5)

            for r in rows:
                row_cells = word_table.add_row().cells
                for i, h in enumerate(headers):
                    row_cells[i].text = str(r[h])
                    for p in row_cells[i].paragraphs:
                        for run in p.runs:
                            run.font.size = Pt(9.5)
            doc.add_paragraph("")

        if item.get("r_code"):
            p_r = doc.add_paragraph("R Kodu:")
            p_r.runs[0].bold = True
            p_code = doc.add_paragraph(item["r_code"])
            for run in p_code.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(9)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=Akademik_APA_Raporu.docx"}
    )


@app.get("/api/export-xlsx")
async def export_apa_excel(session_id: str = "default"):
    history = REPORT_HISTORY.get(session_id, [])
    if not history:
        raise HTTPException(status_code=404, detail="Dışa aktarılacak aktif analiz sonucu bulunamadı.")

    buf = io.BytesIO()
    sheet_counter = 1
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for item in history:
            for tbl in item.get("tables", []):
                rows = tbl.get("rows", [])
                if not rows:
                    continue
                pd.DataFrame(rows).to_excel(writer, sheet_name=f"Analiz_{sheet_counter}", index=False)
                sheet_counter += 1

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=Akademik_Analiz_Sonuclari.xlsx"}
    )


@app.get("/api/export-r")
async def export_r_script(session_id: str = "default"):
    history = REPORT_HISTORY.get(session_id, [])
    if not history:
        raise HTTPException(status_code=404, detail="İndirilecek R betiği bulunamadı.")

    lines = [
        "# Kodsuz Akademik Istatistik & Veri Analizi - Birlesik R Betigi",
        'df <- read.csv("veri_seti.csv", check.names = FALSE)',
        ""
    ]
    for idx, item in enumerate(history, start=1):
        if item.get("r_code"):
            lines.append(f"# ---- {idx}. {item['title']} ----")
            lines.append(item["r_code"])
            lines.append("")

    content = "\n".join(lines).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=Akademik_Analiz_Betigi.R"}
    )