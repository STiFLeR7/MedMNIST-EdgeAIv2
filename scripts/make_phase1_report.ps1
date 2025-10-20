<#
Pretty Phase-1 Report Builder (HAM10000)
- Polished PDF with narrative + tables from CSVs
- Appends figure PDFs (calibration, robustness, pareto)
- Ingests DOCX (summaries) + attempts DOCX->PDF append
- Extracts TensorBoard scalars/images; makes dataset montage
- Auto-opens PDF in VS Code (if available) and default viewer
- **New**: Auto-fit tables to page width + wrapping + header repeat + path shortening
#>

param(
  [string]$Prefix = "ham10000",
  [string]$DocsDir = "D:\MedMNIST-EdgeAIv2Docs",
  [string]$DataRoot = ".\data\HAM10000",
  [string]$OutPdf = ".\HAM10000_Phase1_Report.pdf"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-TemporaryPythonFile {
  [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.IO.Path]::GetRandomFileName() + ".py")
}

function Install-PythonPackages {
  param([string[]]$Pkgs)
  foreach ($p in $Pkgs) {
    $probe = @"
import importlib, sys
try:
    importlib.import_module('$p'); sys.exit(0)
except Exception:
    sys.exit(1)
"@
    $probeFile = New-TemporaryPythonFile
    Set-Content -LiteralPath $probeFile -Value $probe -Encoding UTF8
    & python $probeFile | Out-Null
    $need = $LASTEXITCODE -ne 0
    Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
    if ($need) {
      Write-Host "Installing Python package: $p" -ForegroundColor Yellow
      & python -m pip install --user $p
      if ($LASTEXITCODE -ne 0) { throw "Failed to install Python package: $p" }
    }
  }
}

# --- Path normalization ---
$RepoRoot    = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path "."))
$OutPdfAbs   = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutPdf))
$DocsDirAbs  = [System.IO.Path]::GetFullPath($DocsDir)
$DataRootAbs = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $DataRoot))

# Ensure output dir
$OutDir = [System.IO.Path]::GetDirectoryName($OutPdfAbs)
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = $RepoRoot }
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Pre-escape backslashes for Python raw strings
$RepoRootEsc    = $RepoRoot -replace '\\','\\'
$OutPdfAbsEsc   = $OutPdfAbs -replace '\\','\\'
$DocsDirAbsEsc  = $DocsDirAbs -replace '\\','\\'
$DataRootAbsEsc = $DataRootAbs -replace '\\','\\'

# Ensure Python deps
Install-PythonPackages -Pkgs @(
  "reportlab","pandas","numpy","PyPDF2","python-docx","docx2pdf",
  "tensorboard","matplotlib","Pillow"
)

# --- Embedded Python (auto-fit tables) ---
$py = @"
import os, sys, datetime, math, textwrap
from pathlib import Path
import pandas as pd, numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage, LongTable, KeepInFrame

from PyPDF2 import PdfMerger

# TensorBoard
tb_ok = True
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    tb_ok = False

from PIL import Image

# DOCX ingestion
docx2pdf_available = True
DocxDocument = None
try:
    from docx import Document as DocxDocument
except Exception:
    pass
try:
    from docx2pdf import convert as docx2pdf_convert
except Exception:
    docx2pdf_available = False

# Layout constants
PAGE_W, PAGE_H = A4
MARGIN_L = 1.6*cm; MARGIN_R = 1.6*cm; MARGIN_T = 1.5*cm; MARGIN_B = 1.5*cm
AVAILABLE_W = PAGE_W - (MARGIN_L + MARGIN_R)

ROOT      = Path(r"$RepoRootEsc")
PREFIX    = "$Prefix"
DOCSDIR   = Path(r"$DocsDirAbsEsc")
DATAROOT  = Path(r"$DataRootAbsEsc")
OUTPDF    = Path(r"$OutPdfAbsEsc")
TMP_MAIN  = OUTPDF.with_name(OUTPDF.stem + "_main.pdf")
TMP_DIR   = ROOT / "_report_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
APPEND_LIST = []

# CSV artifacts
core_students = ROOT / 'tables' / 'core' / f'{PREFIX}_students_ci.csv'
core_teacher  = ROOT / 'tables' / 'core' / f'{PREFIX}_teacher_ci.csv'
cal_metrics   = ROOT / 'tables' / 'calibration' / f'{PREFIX}_calibration_metrics.csv'
op_points     = ROOT / 'tables' / 'calibration' / f'{PREFIX}_operating_points.csv'
robust_csv    = ROOT / 'tables' / 'robustness' / f'{PREFIX}_corruptions_ci.csv'
pareto_csv    = ROOT / 'tables' / 'efficiency' / f'{PREFIX}_pareto.csv'
lat_gpu_csv   = ROOT / 'tables' / 'efficiency' / f'{PREFIX}_latency_gpu.csv'
lat_cpu_csv   = ROOT / 'tables' / 'efficiency' / f'{PREFIX}_latency_cpu.csv'
mem_csv       = ROOT / 'tables' / 'efficiency' / f'{PREFIX}_memory.csv'

# PDF figs
calib_dir = ROOT / 'figs' / 'calibration'
robust_fig = ROOT / 'figs' / 'robustness' / f'{PREFIX}_corruption_degradation.pdf'
pareto_fig = ROOT / 'figs' / 'pareto' / f'{PREFIX}_acc_vs_latency.pdf'

def read_csv(p: Path):
    if p.exists():
        try: return pd.read_csv(p)
        except Exception: return None
    return None

# -------- text/paragraph helpers ----------
_styles = getSampleStyleSheet()
def para(txt, size=10, leading=None, bold=False):
    base = _styles['BodyText']
    leading = leading or (size + 2)
    style = ParagraphStyle(
        name=f"Body-{size}-{'Bold' if bold else 'Reg'}",
        parent=base,
        fontName=('Helvetica-Bold' if bold else 'Helvetica'),
        fontSize=size,
        leading=leading,
        wordWrap='CJK'  # allows breaking long tokens
    )
    return Paragraph(txt, style)

def header(txt, lvl=1):
    sizes = {1:16, 2:13, 3:11}
    return para(txt, size=sizes.get(lvl, 11), bold=True, leading=sizes.get(lvl, 11)+2)

def shorten_pathlike(s, keep_parts=3, max_len=60):
    # Keep tail keep_parts of path-ish strings; prepend ellipsis if longer
    if not isinstance(s, str): return s
    if '\\\\' in s or '/' in s:
        parts = s.replace('/', '\\\\').split('\\\\')
        s2 = '\\\\'.join(parts[-keep_parts:])
        if len(s2) > max_len:
            s2 = '…' + s2[-max_len:]
        return s2
    if len(s) > max_len:
        return s[:max_len-1] + '…'
    return s

# -------- table builder with auto-fit ----------
def mk_table(df, round_spec=None, numeric_cols=None, font_size=8, max_rows=100):
    """
    Builds a LongTable that:
      - wraps text via Paragraph
      - shortens path-like cells
      - computes natural column widths and scales to AVAILABLE_W
      - repeats header row on page breaks
    """
    if df is None or df.empty:
        return para("N/A")

    dfx = df.copy()
    if round_spec:
        for k, r in round_spec.items():
            if k in dfx.columns:
                dfx[k] = pd.to_numeric(dfx[k], errors='coerce').round(r)
    if len(dfx) > max_rows:
        dfx = dfx.head(max_rows)

    # Ensure strings and shorten long path-like text
    def fmt_cell(val):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return "-"
        if isinstance(val, (int, float, np.integer, np.floating)):
            return f"{val}"
        return shorten_pathlike(str(val), keep_parts=3, max_len=60)

    cols = list(dfx.columns)
    data_rows = dfx.values.tolist()
    data_rows = [[fmt_cell(v) for v in row] for row in data_rows]

    # Create Paragraph cells for all (wrapping). Numbers stay as strings but wrapped is fine.
    cell_style = ParagraphStyle(
        name="Cell",
        parent=_styles["BodyText"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size+2,
        wordWrap='CJK'
    )
    head_style = ParagraphStyle(
        name="Head",
        parent=_styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size+2,
        wordWrap='CJK'
    )

    head = [Paragraph(str(h), head_style) for h in cols]
    body = [[Paragraph(str(c), cell_style) for c in row] for row in data_rows]
    data = [head] + body

    # Estimate column widths from content using stringWidth on the plain strings
    # then add padding; cap at a min/max and scale to AVAILABLE_W
    min_w = 30  # pt
    max_w = 200 # pt before global scaling
    padd = 10   # pt padding per cell

    raw_widths = []
    for j, col in enumerate(cols):
        # take header + up to N samples
        samples = [str(col)] + [str(r[j]) for r in data_rows[:50]]
        w = max(stringWidth(s, "Helvetica-Bold" if j==j else "Helvetica", font_size) + padd for s in samples)
        w = max(min_w, min(max_w, w))
        raw_widths.append(w)

    total = sum(raw_widths)
    scale = 1.0
    if total > AVAILABLE_W:
        scale = AVAILABLE_W / total
    col_widths = [w*scale for w in raw_widths]

    tbl = LongTable(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('FONT',(0,0),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1), font_size),
        ('BACKGROUND',(0,0),(-1,0),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),0.25,colors.lightgrey),
        ('ALIGN',(0,0),(-1,0),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),3),
        ('RIGHTPADDING',(0,0),(-1,-1),3),
        ('TOPPADDING',(0,0),(-1,-1),2),
        ('BOTTOMPADDING',(0,0),(-1,-1),2),
    ]))
    # Keep table within frame if still a tad too big due to rounding
    return KeepInFrame(AVAILABLE_W, PAGE_H - (MARGIN_T+MARGIN_B), content=[tbl], hAlign='LEFT')

def summarize_core(df):
    if df is None or df.empty: return "No core metrics found."
    metric = None
    for c in ["macro_f1_mean","accuracy_mean","acc_mean"]:
        if c in df.columns: metric = c; break
    if metric is None:
        means = [c for c in df.columns if c.endswith("_mean")]
        if means: metric = means[0]
    if metric:
        best = df.sort_values(metric, ascending=False).head(1)
        try:
            model = str(best["model"].iloc[0]); val = float(best[metric].iloc[0])
            return f"Best student by {metric}: <b>{model}</b> = {val:.3f}."
        except Exception: pass
    return "Core table parsed, metric selection ambiguous."

# --- Dataset montage ---
def build_dataset_montage(root: Path, out_png: Path, n=16, size=128):
    try:
        imgs = sorted(list(root.rglob('*.jpg')) + list(root.rglob('*.png')))
        if not imgs: return None
        take = imgs[:n]; grid = int(math.ceil(math.sqrt(len(take))))
        canvas = Image.new('RGB', (grid*size, grid*size), (255,255,255))
        for idx, p in enumerate(take):
            try:
                im = Image.open(p).convert('RGB').resize((size,size))
                x = (idx % grid)*size; y = (idx // grid)*size
                canvas.paste(im, (x,y))
            except Exception: pass
        canvas.save(out_png); return out_png if out_png.exists() else None
    except Exception: return None

# --- TensorBoard extraction ---
def extract_tb_assets(root_runs: Path, out_dir: Path, label_prefix: str):
    assets = []
    if not tb_ok: return assets
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for ev in sorted(root_runs.rglob('events.*')):
        try:
            ea = EventAccumulator(str(ev), size_guidance={'scalars': 10000, 'tensors': 100, 'images': 50})
            ea.Reload()
            run_name = ev.parent.name
            # Scalars
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            scalar_tags = [t for t in ea.Tags().get('scalars', []) if any(k in t.lower() for k in ['loss','acc','f1','auroc','ap','val'])]
            if scalar_tags:
                plt.figure()
                for t in sorted(scalar_tags):
                    xs = [e.step for e in ea.Scalars(t)]
                    ys = [e.value for e in ea.Scalars(t)]
                    if xs and ys: plt.plot(xs, ys, label=t)
                plt.xlabel('step'); plt.ylabel('value'); plt.title(f'{label_prefix} {run_name} — scalars')
                plt.legend(loc='best', fontsize=6)
                out_png = out_dir / f'{label_prefix}_{run_name}_scalars.png'
                plt.savefig(out_png, bbox_inches='tight', dpi=150); plt.close()
                if out_png.exists(): assets.append(out_png)
            # Images
            for t in ea.Tags().get('images', [])[:3]:
                ims = ea.Images(t)
                if ims:
                    arr = ims[0].encoded_image_string
                    out_png = out_dir / f'{label_prefix}_{run_name}_{t.replace("/","_")}.png'
                    with open(out_png, 'wb') as f: f.write(arr)
                    if out_png.exists(): assets.append(out_png)
        except Exception: pass
    return assets

def build_main_pdf():
    story = []
    # Cover
    story += [
        header('MedMNIST-EdgeAI — Phase-1 (HAM10000)', 1),
        Spacer(1, 0.3*cm),
        para(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'),
        para(f'Repository root: {ROOT}'),
        Spacer(1, 0.5*cm),
        para('This report consolidates teacher/student performance, calibration, operating points, robustness under corruptions, and efficiency (latency/memory). TensorBoard curves/images, dataset montage, and supplementary documents are embedded.', size=10),
        PageBreak()
    ]
    # Roles
    story += [
        header('1. Teacher & Student Roles', 2),
        para('Teacher: high-capacity ResNet-50 trained on HAM10000 at native resolution; provides soft targets for knowledge distillation.', size=10),
        para('Students: ResNet-18, MobileNetV2, EfficientNet-B0 distilled with KD (T, α) and optional Attention Transfer (λ), evaluated over multiple seeds.', size=10),
        Spacer(1, 0.3*cm)
    ]
    # Dataset montage
    montage = build_dataset_montage(DATAROOT, TMP_DIR / 'dataset_montage.png', n=16, size=128)
    if montage and montage.exists():
        story += [header('Dataset Visual Snapshot', 3), RLImage(str(montage), width=14*cm, height=14*cm), Spacer(1,0.3*cm)]
    story += [PageBreak()]

    # Load CSVs
    def read_csv_safe(p): 
        try: 
            return pd.read_csv(p) if p.exists() else None
        except Exception: 
            return None
    df_teacher  = read_csv_safe(core_teacher)
    df_students = read_csv_safe(core_students)
    df_cal      = read_csv_safe(cal_metrics)
    df_op       = read_csv_safe(op_points)
    df_rob      = read_csv_safe(robust_csv)
    df_pareto   = read_csv_safe(pareto_csv)
    df_latg     = read_csv_safe(lat_gpu_csv)
    df_latc     = read_csv_safe(lat_cpu_csv)
    df_mem      = read_csv_safe(mem_csv)

    # Performance
    story += [header('2. Performance (Test-set with CI)', 2)]
    if df_teacher is not None:
        story += [para('Teacher summary:'), mk_table(df_teacher, round_spec={'macro_f1_mean':3,'accuracy_mean':3}, font_size=8), Spacer(1,0.2*cm)]
    if df_students is not None:
        story += [para('Students summary:'), mk_table(df_students, round_spec={'macro_f1_mean':3,'accuracy_mean':3}, font_size=8), Spacer(1,0.2*cm)]
        story += [para(summarize_core(df_students))]
    story += [PageBreak()]

    # Calibration
    story += [header('3. Calibration & Operating Points', 2)]
    if df_cal is not None:
        story += [para('Calibration metrics:'), mk_table(df_cal, round_spec={'ece':3,'nll':3,'brier':3,'ece_adaptive':3}, font_size=8), Spacer(1,0.2*cm)]
    if df_op is not None:
        story += [para('Operating points (per model/seed):'), mk_table(df_op, round_spec={'threshold':3,'tau':3,'macro_f1_opt':3}, font_size=8), Spacer(1,0.2*cm)]
    story += [PageBreak()]

    # Robustness
    story += [header('4. Robustness under Corruptions', 2)]
    if df_rob is not None:
        story += [para('Macro-F1 across corruption levels:'), mk_table(df_rob, round_spec={'macro_f1':3}, font_size=8), Spacer(1,0.2*cm)]
    else:
        story += [para('No robustness CSV found.')]

    # Efficiency
    story += [header('5. Efficiency (Latency & Memory)', 2)]
    if df_latg is not None:
        story += [para('GPU latency (aggregated):'), mk_table(df_latg, round_spec={'lat_ms_mean':1,'lat_ms_std':3,'lat_ms_p50':1}, font_size=8), Spacer(1,0.2*cm)]
    if df_latc is not None:
        story += [para('CPU latency (aggregated):'), mk_table(df_latc, round_spec={'lat_ms_mean':1,'lat_ms_std':3,'lat_ms_p50':1}, font_size=8), Spacer(1,0.2*cm)]
    if df_mem is not None:
        story += [para('Model memory footprint:'), mk_table(df_mem, round_spec={'params_mib':1,'peak_cuda_mib':1}, font_size=8), Spacer(1,0.2*cm)]

    # Pareto
    story += [header('6. Pareto (Accuracy vs Latency)', 2)]
    if df_pareto is not None:
        story += [mk_table(df_pareto, round_spec={'score':3,'latency_ms':1,'params_mib':1,'peak_cuda_mib':1}, font_size=8), Spacer(1,0.2*cm)]
    story += [para('See appended figure: Acc-vs-Latency bubble plot.')]

    # TensorBoard
    story += [PageBreak(), header('7. TensorBoard Curves & Images', 2)]
    tb_assets = []
    tb_assets += extract_tb_assets(ROOT / 'models' / 'teachers', TMP_DIR, 'teacher')
    tb_assets += extract_tb_assets(ROOT / 'models' / 'students', TMP_DIR, 'student')
    if tb_assets:
        for p in tb_assets:
            story += [RLImage(str(p), width=15*cm, height=9*cm), Spacer(1,0.2*cm)]
    else:
        story += [para('No TensorBoard event assets found or TensorBoard parser unavailable.')]

    # DOCX summaries
    story += [PageBreak(), header('8. Supplementary Notes from DOCX', 2)]
    docx_files = sorted(DOCSDIR.glob('*.docx')) if DOCSDIR.exists() else []
    if not docx_files:
        story += [para('No .docx files found in the supplementary folder.')]
    else:
        for p in docx_files:
            story += [header(f'- {p.name}', 3)]
            txt = None
            try:
                if DocxDocument is not None:
                    d = DocxDocument(str(p))
                    paras = [q.text.strip() for q in d.paragraphs if q.text.strip()]
                    text = '\n'.join(paras)
                    if len(text) > 4000: text = text[:4000] + ' ... [truncated]'
                    txt = text
            except Exception:
                pass
            if txt:
                for chunk in textwrap.wrap(txt, width=120):
                    story += [para(chunk, size=9)]
                story += [Spacer(1,0.3*cm)]
            else:
                story += [para('(Could not extract text.)')]

    # Build main report
    TMP_MAIN.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(TMP_MAIN), pagesize=A4,
                            leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                            topMargin=MARGIN_T, bottomMargin=MARGIN_B)
    doc.build(story)

def append_fig_pdfs():
    if calib_dir.exists():
        for p in sorted(calib_dir.glob('*.pdf')): APPEND_LIST.append(p)
    for p in [robust_fig, pareto_fig]:
        if p.exists(): APPEND_LIST.append(p)

def convert_and_append_docx_pdfs():
    if not DOCSDIR.exists(): return
    for docx in sorted(DOCSDIR.glob('*.docx')):
        out_pdf = OUTPDF.with_name(OUTPDF.stem + f'__{docx.stem}.pdf')
        if docx2pdf_available:
            try:
                out_pdf.parent.mkdir(parents=True, exist_ok=True)
                docx2pdf_convert(str(docx), str(out_pdf))
                if out_pdf.exists(): APPEND_LIST.append(out_pdf)
            except Exception: pass

def merge_all():
    merger = PdfMerger()
    merger.append(str(TMP_MAIN))
    for p in APPEND_LIST:
        try: merger.append(str(p))
        except Exception: pass
    OUTPDF.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPDF,'wb') as f: merger.write(f)
    merger.close()

def main():
    build_main_pdf()
    append_fig_pdfs()
    convert_and_append_docx_pdfs()
    merge_all()
    print(f'Wrote {OUTPDF}')

if __name__ == '__main__':
    main()
"@

# Run Python
$tmpPy = New-TemporaryPythonFile
Set-Content -LiteralPath $tmpPy -Value $py -Encoding UTF8
& python $tmpPy
$exit = $LASTEXITCODE
Remove-Item -LiteralPath $tmpPy -Force -ErrorAction SilentlyContinue
if ($exit -ne 0) { throw "Report build failed (python exit $exit)." }

# Open in VS Code (if available) and default viewer
try {
  $codeCmd = Get-Command -Name "code" -ErrorAction SilentlyContinue
  if ($null -ne $codeCmd) {
    Start-Process -FilePath $codeCmd.Source -ArgumentList ("-r `"{0}`"" -f $OutPdfAbs) | Out-Null
  }
} catch { }
Start-Process -FilePath $OutPdfAbs | Out-Null
Write-Host "Report ready: $OutPdfAbs" -ForegroundColor Green
