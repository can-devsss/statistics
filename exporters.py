"""
Rapor dışa aktarma (Word / Excel / R) — main.py'den bağımsız, yalnızca python-docx ve openpyxl'e dayanır.

Tasarım ilkeleri
- Girdi her zaman "history" listesidir: [{title, apa_note, notes, r_code, tables:[{title, rows:[{...}]}]}]
- XML'de geçersiz kontrol karakterleri temizlenir (python-docx / openpyxl aksi halde hata verir).
- NaN / inf / None güvenli şekilde "-" olur.
- Her fonksiyon doğrudan `bytes` döndürür (Content-Length doğru gider, yarım dosya inmez).
"""
import io
import math
import re
from typing import Any, Dict, List

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# XML 1.0'da geçersiz karakterler
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]")
_P_COLUMNS = {"p", "p-adj", "model p", "trend p", "mk p", "pr > f", "levene p", "pearson p", "spearman p"}
_DECIMAL = re.compile(r"^-?\d+(\.\d+)?$")

MAX_REPORTS = 300
MAX_TABLES_PER_REPORT = 50
MAX_ROWS_PER_TABLE = 5000
MAX_COLS_PER_TABLE = 60


def clean_text(value: Any) -> str:
    """Her değeri XML-güvenli düz metne çevirir."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.10g}"
    text = str(value)
    text = text.encode("utf-8", "ignore").decode("utf-8")  # yalnız (lone) surrogate'ları at
    return _ILLEGAL_XML.sub("", text)


def normalize_history(history: Any) -> List[Dict[str, Any]]:
    """İstemciden gelen / sunucuda tutulan geçmişi doğrular ve güvenli biçime getirir."""
    if not isinstance(history, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in history[:MAX_REPORTS]:
        if not isinstance(item, dict):
            continue
        tables = []
        for tbl in (item.get("tables") or [])[:MAX_TABLES_PER_REPORT]:
            if not isinstance(tbl, dict):
                continue
            rows = [r for r in (tbl.get("rows") or [])[:MAX_ROWS_PER_TABLE] if isinstance(r, dict)]
            if not rows:
                continue
            tables.append({"title": clean_text(tbl.get("title", "")), "rows": rows})
        notes = item.get("notes") or []
        if not isinstance(notes, list):
            notes = [notes]
        out.append({
            "title": clean_text(item.get("title", "Analiz")),
            "apa_note": clean_text(item.get("apa_note", "")) if item.get("apa_note") else "",
            "notes": [clean_text(n) for n in notes if n],
            "r_code": clean_text(item.get("r_code", "")) if item.get("r_code") else "",
            "tables": tables,
        })
    return out


def _headers(rows: List[Dict[str, Any]]) -> List[str]:
    """Tüm satırlardaki anahtarların birleşimi (ilk görülme sırasıyla)."""
    seen: Dict[str, None] = {}
    for r in rows:
        for k in r.keys():
            seen.setdefault(str(k), None)
    return list(seen.keys())[:MAX_COLS_PER_TABLE]


def _cell_value(row: Dict[str, Any], header: str) -> Any:
    if header in row:
        return row[header]
    for k, v in row.items():  # anahtar int olabilir
        if str(k) == header:
            return v
    return None


# ------------------------------------------------------------------ WORD
def _set_font(run_or_style, name: str, size: float = None) -> None:
    run_or_style.font.name = name
    if size:
        run_or_style.font.size = Pt(size)
    el = run_or_style.element if hasattr(run_or_style, "element") else None
    rpr = None
    if el is not None:
        rpr = el.get_or_add_rPr() if hasattr(el, "get_or_add_rPr") else getattr(el, "rPr", None)
    if rpr is not None:
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(attr), name)


def _shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _row_flags(row, header: bool = False) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)
    if header:
        th = OxmlElement("w:tblHeader")
        tr_pr.append(th)


def build_docx(history: List[Dict[str, Any]]) -> bytes:
    history = normalize_history(history)
    doc = Document()

    # Yatay sayfa: geniş tablolar (12+ sütun) taşmasın
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    if sec.page_width < sec.page_height:
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = sec.bottom_margin = Cm(1.8)

    # Varsayılan yazı tipi: Times New Roman (APA) — East-Asian font ataması da açıkça yapılır
    normal = doc.styles["Normal"]
    _set_font(normal, "Times New Roman", 11)
    rpr = normal.element.get_or_add_rPr()
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "tr-TR")
    rpr.append(lang)

    h = doc.add_heading("Akademik İstatistik & Veri Analizi Raporu (APA)", level=1)
    for r in h.runs:
        _set_font(r, "Times New Roman")

    if not history:
        doc.add_paragraph("Raporlanacak analiz bulunamadı.")

    for idx, item in enumerate(history, start=1):
        h2 = doc.add_heading(f"{idx}. {item['title']}", level=2)
        for r in h2.runs:
            _set_font(r, "Times New Roman")

        if item["apa_note"]:
            p = doc.add_paragraph()
            run = p.add_run(item["apa_note"])
            run.italic = True

        for note in item["notes"]:
            p = doc.add_paragraph()
            run = p.add_run("Not: ")
            run.bold = True
            p.add_run(note)

        for tbl in item["tables"]:
            rows = tbl["rows"]
            headers = _headers(rows)
            if not headers:
                continue
            p_title = doc.add_paragraph()
            rt = p_title.add_run(tbl["title"])
            rt.bold = True
            p_title.paragraph_format.keep_with_next = True

            font_pt = 9 if len(headers) > 8 else 10
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = "Table Grid"
            table.autofit = True

            hdr_row = table.rows[0]
            _row_flags(hdr_row, header=True)
            for i, head in enumerate(headers):
                cell = hdr_row.cells[i]
                cell.text = ""
                run = cell.paragraphs[0].add_run(clean_text(head))
                run.bold = True
                run.font.size = Pt(font_pt)
                _shade(cell, "E7E9EE")

            for r in rows:
                new_row = table.add_row()
                _row_flags(new_row)
                for i, head in enumerate(headers):
                    cell = new_row.cells[i]
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(clean_text(_cell_value(r, head)))
                    run.font.size = Pt(font_pt)
            doc.add_paragraph("")

        if item["r_code"]:
            p_r = doc.add_paragraph()
            p_r.add_run("R Kodu:").bold = True
            p_code = doc.add_paragraph()
            run = p_code.add_run(item["r_code"])
            _set_font(run, "Consolas", 9)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------ EXCEL
_BAD_SHEET_CHARS = re.compile(r"[\[\]\:\*\?\/\\]")


def _sheet_name(base: str, used: set) -> str:
    name = _BAD_SHEET_CHARS.sub(" ", clean_text(base)).strip().strip("'")
    name = re.sub(r"\s+", " ", name) or "Sayfa"
    name = name[:31].rstrip().rstrip("'") or "Sayfa"
    candidate, n = name, 2
    while candidate.lower() in used:
        suffix = f" ({n})"
        candidate = name[: 31 - len(suffix)] + suffix
        n += 1
    used.add(candidate.lower())
    return candidate


def _write_cell(ws, row: int, col: int, value: Any, header: str = "") -> None:
    cell = ws.cell(row=row, column=col)
    if isinstance(value, bool):
        cell.value = "Evet" if value else "Hayır"
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            cell.value = "-"
        else:
            cell.value = value
        return
    text = clean_text(value)
    if header.lower() in _P_COLUMNS and _DECIMAL.match(text):
        cell.value = float(text)  # p değerleri gerçek sayı olsun (Excel'de "metin olarak sayı" uyarısı çıkmasın)
        cell.number_format = "0.0000"
        return
    cell.value = text
    if text.startswith("="):  # formül enjeksiyonunu engelle
        cell.data_type = "s"


def build_xlsx(history: List[Dict[str, Any]]) -> bytes:
    history = normalize_history(history)
    wb = Workbook()
    head_font = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="E7E9EE")
    thin = Side(style="thin", color="BFC3CC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    used: set = set()

    ws0 = wb.active
    ws0.title = _sheet_name("Özet", used)
    ws0.append(["No", "Analiz", "Tablo Sayısı", "APA Bulgu Özeti", "Varsayım Notları"])
    for c in ws0[1]:
        c.font = head_font
        c.fill = head_fill
        c.border = border
    for idx, item in enumerate(history, start=1):
        ws0.append([idx, item["title"], len(item["tables"]), item["apa_note"], " ".join(item["notes"])])
    for row in ws0.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if isinstance(c.value, str) and c.value.startswith("="):
                c.data_type = "s"
    for col, width in zip("ABCDE", (6, 34, 14, 90, 60)):
        ws0.column_dimensions[col].width = width
    ws0.freeze_panes = "A2"

    for idx, item in enumerate(history, start=1):
        for t_i, tbl in enumerate(item["tables"], start=1):
            rows = tbl["rows"]
            headers = _headers(rows)
            if not headers:
                continue
            ws = wb.create_sheet(_sheet_name(f"{idx}.{t_i} {tbl['title']}", used))
            ws.cell(row=1, column=1, value=tbl["title"]).font = Font(bold=True, size=12)
            ws.cell(row=2, column=1, value=f"{idx}. {item['title']}").font = Font(italic=True, color="666666")

            hdr_row = 4
            for ci, head in enumerate(headers, start=1):
                c = ws.cell(row=hdr_row, column=ci, value=clean_text(head))
                c.font, c.fill, c.border = head_font, head_fill, border
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            widths = [len(clean_text(h)) for h in headers]
            for ri, r in enumerate(rows, start=hdr_row + 1):
                for ci, head in enumerate(headers, start=1):
                    val = _cell_value(r, head)
                    _write_cell(ws, ri, ci, val, head)
                    ws.cell(row=ri, column=ci).border = border
                    widths[ci - 1] = max(widths[ci - 1], len(clean_text(val)))
            for ci, w in enumerate(widths, start=1):
                ws.column_dimensions[get_column_letter(ci)].width = min(max(10, w + 3), 60)
            ws.freeze_panes = ws.cell(row=hdr_row + 1, column=1)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------ R
def build_r_script(history: List[Dict[str, Any]]) -> bytes:
    history = normalize_history(history)
    lines = [
        "# -*- encoding: UTF-8 -*-",
        "# Kodsuz Akademik Istatistik & Veri Analizi - Birlesik R Betigi",
        "# Not: Turkce karakterler icin dosyayi source('betik.R', encoding = 'UTF-8') ile calistirin.",
        'library(survival)  # yalniz sagkalim/Cox analizleri icin gerekir',
        'df <- read.csv("veri_seti.csv", check.names = FALSE, fileEncoding = "UTF-8")',
        "",
    ]
    for idx, item in enumerate(history, start=1):
        if item["r_code"]:
            lines.append(f"# ---- {idx}. {item['title']} ----")
            lines.append(item["r_code"])
            lines.append("")
    return "\n".join(lines).encode("utf-8")
