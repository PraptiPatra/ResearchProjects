"""
app.py
------
FastAPI app serving the UI and endpoints: /, /images, /terrain, /routes
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, math
import numpy as np
from pathlib import Path

from .routing import composite_cost, astar, score_path, color_from_metrics, propose_stops

BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / 'data'
APPDIR = BASE / 'app'
STATIC = APPDIR / 'static'

app = FastAPI(title="Lunar Navigation Planner")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

def find_annotation_file() -> Path | None:
    dataset_root = BASE / 'dataset'
    if not dataset_root.exists(): return None
    for path in dataset_root.rglob('*_annotations.coco.json'):
        return path
    return None

_ANN_PATH = find_annotation_file()
if _ANN_PATH is None:
    print("⚠️ No COCO annotation JSON found – /images will be empty until you add it.")

@app.get("/")
def home():
    index_path = STATIC / 'index.html'
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))

@app.get("/images")
def images():
    if _ANN_PATH is None:
        return {'images': []}
    with open(_ANN_PATH, 'r') as f:
        coco = json.load(f)
    out = []
    for img in coco.get('images', []):
        img_id = img['id']; W, H = img['width'], img['height']
        hazard_path = DATA / f"{img_id}_hazard.npy"
        if hazard_path.exists():
            hazards = np.load(hazard_path)
            frac = float((hazards > 0.5).sum() / hazards.size)
        else:
            frac = 0.0
        out.append({
            'image_id': img_id,
            'file': img['file_name'],
            'width': W, 'height': H,
            'hazard_fraction': round(frac, 4),
            'suggest_x': W // 2, 'suggest_y': H // 2
        })
    out.sort(key=lambda x: x['hazard_fraction'])
    return {'images': out}

@app.get("/terrain")
def terrain(image_id: int):
    dem    = DATA / f"{image_id}_dem.npy"
    hazard = DATA / f"{image_id}_hazard.npy"
    illum  = DATA / f"{image_id}_illum.npy"
    if not dem.exists() or not hazard.exists() or not illum.exists():
        raise HTTPException(status_code=404, detail=f"Data for image {image_id} not found")
    dem_a = np.load(dem).astype(np.float32)
    haz_a = np.load(hazard).astype(np.float32)
    ill_a = np.load(illum).astype(np.float32)
    return {
        'shape': [int(dem_a.shape[0]), int(dem_a.shape[1])],
        'meters_per_pixel': 1.0,
        'dem': dem_a.tolist(),
        'hazards': haz_a.tolist(),
        'illum': ill_a.tolist(),
    }

class RouteRequest(BaseModel):
    image_id: int
    landing_xy: tuple[int,int]
    min_length_m: float = 100.0
    meters_per_pixel: float = 1.0
    top_k: int = 5

@app.post("/routes")
def routes(req: RouteRequest):
    image_id = req.image_id
    dem    = DATA / f"{image_id}_dem.npy"
    hazard = DATA / f"{image_id}_hazard.npy"
    illum  = DATA / f"{image_id}_illum.npy"
    if not dem.exists() or not hazard.exists() or not illum.exists():
        raise HTTPException(status_code=404, detail=f"Data for image {image_id} not found")

    dem_a = np.load(dem).astype(np.float32)
    haz_a = np.load(hazard).astype(np.float32)
    ill_a = np.load(illum).astype(np.float32)

    gy, gx = np.gradient(dem_a)
    slope  = np.sqrt(gx*gx + gy*gy)
    slope  = (slope - slope.min())/(slope.max() - slope.min() + 1e-6)

    cost   = composite_cost(slope, haz_a, ill_a)
    H, W   = dem_a.shape
    sx, sy = int(req.landing_xy[0]), int(req.landing_xy[1])
    sx = max(0, min(W-1, sx)); sy = max(0, min(H-1, sy))

    mpp = max(req.meters_per_pixel, 1e-3)
    r   = max(int(req.min_length_m / mpp), 20)
    goals=[]
    for th in np.linspace(0, 2*math.pi, 36, endpoint=False):
        gx_ = sx + int(r*np.cos(th)); gy_ = sy + int(r*np.sin(th))
        if 0<=gx_<W and 0<=gy_<H: goals.append((gx_,gy_))

    candidates=[]
    for gx_, gy_ in goals:
        path = astar(cost, (sx,sy), (gx_,gy_))
        if not path: continue
        length_m = (len(path)-1)*mpp
        if length_m < req.min_length_m: continue
        sc, expl = score_path(path, slope, haz_a, ill_a)
        candidates.append({'path':path, 'length_m':float(length_m), 'score':sc, 'explain':expl})

    candidates.sort(key=lambda c: c['score'])
    top = candidates[:req.top_k]
    routes_out=[]
    albedo = (dem_a - dem_a.min())/(dem_a.max() - dem_a.min() + 1e-6) * 255.0
    for c in top:
        stops = propose_stops(c['path'], albedo, ill_a, dem_a, n=10)
        color = color_from_metrics(c['explain'])
        s,h,d = c['explain']['avg_slope_norm'], c['explain']['hazard_density'], c['explain']['shadow_factor']
        pros, cons = [], []
        pros.append('Gentle slopes' if s < 0.25 else None); 
        if s >= 0.25: cons.append('Moderate slopes — reduce speed')
        pros.append('Sparse hazards' if h < 0.05 else None);
        if h >= 0.05: cons.append('Hazard pockets — detours needed')
        pros.append('Good solar exposure' if d < 0.35 else None);
        if d >= 0.35: cons.append('Shadow stretches — plan timing')
        pros = [p for p in pros if p]
        routes_out.append({
            'color': color,
            'length_m': c['length_m'],
            'score': c['score'],
            'path': [{'x':int(x),'y':int(y)} for x,y in c['path']],
            'stops': stops,
            'pros': pros,
            'cons': cons
        })
    return {'routes': routes_out, 'landing_xy': [sx,sy], 'meters_per_pixel': mpp}
