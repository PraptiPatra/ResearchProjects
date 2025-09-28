"""
routing.py
---------------
Core helpers for lunar route planning. These functions combine slope, hazard
and illumination into a cost map, perform A* search to find routes, score
those routes, assign colours based on their quality, and propose stops.
"""

import numpy as np
import heapq

def composite_cost(slope: np.ndarray, hazards: np.ndarray, illum: np.ndarray,
                   w_slope: float = 0.55, w_hazard: float = 0.30,
                   w_dark: float = 0.15) -> np.ndarray:
    s = np.clip(slope,   0.0, 1.0)
    h = np.clip(hazards, 0.0, 1.0)
    d = 1.0 - np.clip(illum, 0.0, 1.0)
    return w_slope * s + w_hazard * h + w_dark * d

def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    H, W = cost.shape
    sx, sy = start; gx, gy = goal
    sx = int(np.clip(sx, 0, W - 1)); sy = int(np.clip(sy, 0, H - 1))
    gx = int(np.clip(gx, 0, W - 1)); gy = int(np.clip(gy, 0, H - 1))
    def h(x: int, y: int) -> float: return abs(x - gx) + abs(y - gy)
    open_set = [(h(sx, sy), 0.0, (sx, sy), None)]
    came_from = {}; gscore = {(sx, sy): 0.0}
    neighbours = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)]
    while open_set:
        _, g, (x,y), parent = heapq.heappop(open_set)
        if (x,y) in came_from: continue
        came_from[(x,y)] = parent
        if (x,y) == (gx,gy):
            path = [(x,y)]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse(); return path
        for dx,dy in neighbours:
            nx, ny = x+dx, y+dy
            if nx<0 or nx>=W or ny<0 or ny>=H: continue
            step_cost = float(cost[ny,nx]) + 1e-6
            cand = g + step_cost
            if (nx,ny) not in gscore or cand < gscore[(nx,ny)]:
                gscore[(nx,ny)] = cand
                heapq.heappush(open_set, (cand + h(nx,ny), cand, (nx,ny), (x,y)))
    return None

def score_path(path: list[tuple[int,int]], slope: np.ndarray,
               hazards: np.ndarray, illum: np.ndarray):
    if not path: return 1e9, {}
    px = np.array([p[0] for p in path], dtype=np.int64)
    py = np.array([p[1] for p in path], dtype=np.int64)
    s = slope[py, px]; h = hazards[py, px]; d = 1.0 - illum[py, px]
    total = 0.55*s.mean() + 0.3*h.mean() + 0.15*d.mean()
    return float(total), {
        'avg_slope_norm': float(s.mean()),
        'hazard_density': float(h.mean()),
        'shadow_factor': float(d.mean())
    }

def color_from_metrics(metrics: dict) -> str:
    s, h, d = metrics['avg_slope_norm'], metrics['hazard_density'], metrics['shadow_factor']
    if (s < 0.25) and (h < 0.05) and (d < 0.35): return 'green'
    if (s < 0.45) and (h < 0.12) and (d < 0.55): return 'yellow'
    return 'red'

def propose_stops(path: list[tuple[int,int]], albedo: np.ndarray,
                  illum: np.ndarray, dem: np.ndarray, n: int = 10) -> list[dict]:
    if not path: return []
    idxs = np.linspace(0, len(path)-1, n+2).astype(int)[1:-1]
    stops = []
    for i in idxs:
        x,y = path[i]
        a_val = float(albedo[y,x]); l_val = float(illum[y,x])
        local = dem[max(0,y-1):y+2, max(0,x-1):x+2]
        rough = float(np.std(local)) if local.size>0 else 0.0
        notes=[]
        if l_val>0.75: notes.append('High illumination (good for solar & imaging)')
        if l_val<0.35: notes.append('Low illumination (thermal/volatile interest)')
        if rough<8.0: notes.append('Smooth patch (sampling)')
        if a_val>170: notes.append('High-albedo anomaly (composition)')
        if a_val<90:  notes.append('Low-albedo patch (space weathering)')
        stops.append({'x':int(x),'y':int(y),'illum':l_val,'albedo':a_val,
                      'note':'; '.join(notes) or 'General observation'})
    return stops
