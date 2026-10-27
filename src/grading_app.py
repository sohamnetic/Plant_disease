"""Local severity grading tool (human review of outputs/severity/severity_grades.csv).

Shows each original photo next to its measured-lesion overlay (Phase 5; pink =
affected tissue) with the current grade, the measured % and where the grade came
from. A grader presses 0-4 to confirm or overwrite; every save writes the CSV
immediately with grade_source="human". Bacterial spot rot grades are by-eye (the
measurement misses dark rot) and benefit most from an expert.

  python -m uvicorn src.grading_app:app --port 8010
  open http://127.0.0.1:8010/   (keys: 0-4 grade, <- / -> navigate, h = only not-yet-human)

After grading, refit the severity head (diagnosis unchanged):
  python src/refit_severity.py && python src/calibrate_v2.py --exp-dir experiments/experiment_006_v3_measured_severity
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

GRADES_CSV = cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv"
LOCK = threading.Lock()
app = FastAPI(title="Severity grading")


def _load() -> pd.DataFrame:
    return pd.read_csv(GRADES_CSV)


@app.get("/items")
def items() -> list[dict]:
    df = _load()
    cols = ["filepath", "class", "split", "grade", "grade_source", "measured_pct", "grade_visual"]
    return df[[c for c in cols if c in df.columns]].reset_index().fillna("").to_dict("records")


@app.get("/overlay/{i}")
def overlay_image(i: int) -> FileResponse:
    from measure_severity import overlay_path

    df = _load()
    if not 0 <= i < len(df):
        raise HTTPException(404)
    path = overlay_path(df.loc[i, "filepath"])
    if not path.is_file():
        raise HTTPException(404, "no overlay: run python src/measure_severity.py measure")
    return FileResponse(path)


@app.get("/image/{i}")
def image(i: int) -> FileResponse:
    df = _load()
    if not 0 <= i < len(df):
        raise HTTPException(404)
    return FileResponse(cfg.PROJECT_ROOT / df.loc[i, "filepath"])


@app.post("/grade/{i}/{g}")
def grade(i: int, g: int) -> dict:
    if g not in range(5):
        raise HTTPException(400, "grade must be 0-4")
    with LOCK:
        df = _load()
        if not 0 <= i < len(df):
            raise HTTPException(404)
        df.loc[i, "grade"] = g
        df.loc[i, "grade_source"] = "human"
        df.to_csv(GRADES_CSV, index=False)
    return {"index": i, "grade": g}


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/><title>Severity grading</title>
<style>
body{font-family:Segoe UI,sans-serif;margin:0;background:#111;color:#eee;display:flex;flex-direction:column;height:100vh}
header{padding:8px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
main{flex:1;display:flex;justify-content:center;align-items:center;gap:8px;min-height:0}
img{max-width:50%;max-height:100%;object-fit:contain}
button{font-size:16px;padding:8px 14px;margin:2px;border:0;border-radius:6px;cursor:pointer}
button.on{background:#2e7d32;color:#fff}
.meta{opacity:.8}
</style></head><body>
<header>
 <b id="pos"></b><span class="meta" id="meta"></span>
 <span id="btns"></span>
 <label><input type="checkbox" id="onlyTodo"/> only not-yet-human</label>
</header>
<main><img id="img"/><img id="ov" onerror="this.style.display='none'" onload="this.style.display=''"/></main>
<footer style="padding:6px 16px" class="meta">% of the worst leaf/head affected: 0 none · 1 early (&lt;5% or a few small spots) · 2 moderate (5-25%) · 3 severe (25-50%) · 4 critical (&gt;50% / collapse). Right image: pink = measured affected tissue.</footer>
<script>
let items=[],i=0;
const labels=["0 none","1 early","2 moderate","3 severe","4 critical"];
async function load(){items=await (await fetch('/items')).json();show();}
function visible(){const t=document.getElementById('onlyTodo').checked;return items.filter(x=>!t||x.grade_source!=='human');}
function show(){const v=visible();if(!v.length){document.getElementById('pos').textContent='done';return;}
 i=Math.max(0,Math.min(i,v.length-1));const it=v[i];
 document.getElementById('img').src='/image/'+it.index;
 document.getElementById('ov').src='/overlay/'+it.index;
 document.getElementById('pos').textContent=(i+1)+' / '+v.length;
 document.getElementById('meta').textContent=it['class']+' · '+it.split+' · measured '+(it.measured_pct===''?'-':it.measured_pct+'%')+' · earlier by-eye grade '+it.grade_visual+' · source='+it.grade_source;
 document.getElementById('btns').innerHTML=labels.map((l,g)=>`<button class="${g===it.grade?'on':''}" onclick="setG(${g})">${l}</button>`).join('');}
async function setG(g){const v=visible();const it=v[i];await fetch(`/grade/${it.index}/${g}`,{method:'POST'});
 it.grade=g;it.grade_source='human';i++;show();}
document.addEventListener('keydown',e=>{if(e.key>='0'&&e.key<='4')setG(+e.key);
 if(e.key==='ArrowRight'){i++;show();} if(e.key==='ArrowLeft'){i--;show();}
 if(e.key==='h'){const c=document.getElementById('onlyTodo');c.checked=!c.checked;i=0;show();}});
document.getElementById('onlyTodo').onchange=()=>{i=0;show();};
load();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE
