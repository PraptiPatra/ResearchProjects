import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
Ky = Kx.T

def conv2(a, k):
    H, W = a.shape
    kh, kw = k.shape
    ph, pw = kh//2, kw//2
    ap = np.pad(a, ((ph,ph),(pw,pw)), mode='edge')
    out = np.empty_like(a, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            out[y,x] = (ap[y:y+kh, x:x+kw] * k).sum()
    return out

def polygons_or_bbox(coco, image_id):
    polys=[]
    for ann in coco["annotations"]:
        if ann["image_id"] != image_id: continue
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            for s in seg:
                if len(s) >= 6:
                    polys.append([(s[i], s[i+1]) for i in range(0, len(s), 2)])
        else:
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                x,y,w,h = bbox
                polys.append([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
    return polys

def main(args):
    coco_path = Path(args.coco).resolve()
    out_dir   = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, 'r') as f:
        coco = json.load(f)

    SUN_AZIMUTH_DEG = 135.0
    theta = math.radians(SUN_AZIMUTH_DEG)
    dirx, diry = math.cos(theta), math.sin(theta)

    for img in coco["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        img_path = (coco_path.parent / file_name).resolve()

        gray = Image.open(img_path).convert("L")
        W,H  = gray.size
        gray_blur = np.array(gray.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)

        # Hazard mask
        mask = Image.new("L", (W,H), 0)
        draw = ImageDraw.Draw(mask)
        for pts in polygons_or_bbox(coco, img_id):
            draw.polygon(pts, outline=255, fill=255)
        hazard = np.array(mask, dtype=np.uint8) / 255.0

        # Slope via Sobel
        gx = conv2(gray_blur, Kx)
        gy = conv2(gray_blur, Ky)
        slope = np.sqrt(gx**2 + gy**2)
        slope = (slope - slope.min())/(slope.max() - slope.min() + 1e-6)

        # Illumination proxy
        illum = 0.5 + 0.5 * ((gx*dirx + gy*diry) / (np.sqrt(gx*gx + gy*gy) + 1e-6))
        illum = np.nan_to_num(illum, nan=0.5)
        illum = (illum - illum.min())/(illum.max() - illum.min() + 1e-6)

        dem = gray_blur.copy()

        np.save(out_dir / f"{img_id}_hazard.npy", hazard)
        np.save(out_dir / f"{img_id}_slope.npy",  slope)
        np.save(out_dir / f"{img_id}_illum.npy",  illum)
        np.save(out_dir / f"{img_id}_dem.npy",    dem)

    print("✅ Saved .npy files to:", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="Path to *_annotations.coco.json")
    p.add_argument("--out",  default="lunar_arvr/data", help="Output folder for .npy files")
    main(p.parse_args())
