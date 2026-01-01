#!/usr/bin/env python3
import argparse, json
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np, pandas as pd

@dataclass(frozen=True)
class Knob:
    stream: int; model: str; QP: int; FPS: int; Resolution: str

class CurveEstimator:
    def __init__(self, df: pd.DataFrame, metric_col: str):
        self.metric_col = metric_col
        self.curves: Dict[Tuple[int,str,int,int,str], Tuple[np.ndarray,np.ndarray]] = {}
        req = {"stream","model","QP","FPS","Resolution","Bitrate",metric_col}
        miss = req - set(df.columns)
        if miss: raise ValueError(f"Missing columns: {miss}")
        for key, g in df.groupby(["stream","model","QP","FPS","Resolution"]):
            agg = g.groupby("Bitrate", as_index=False)[metric_col].mean().sort_values("Bitrate")
            br  = agg["Bitrate"].to_numpy(float)
            met = agg[metric_col].to_numpy(float)
            if len(br) >= 2: self.curves[key] = (br, met)

    def predict(self, k: Knob, b: float) -> float:
        key = (k.stream, k.model, k.QP, k.FPS, k.Resolution)
        if key not in self.curves:
            raise KeyError(f"No curve for {key}")
        br, met = self.curves[key]

        # S A T U R A S I  (tanpa extrapolation linear)
        if b <= br[0]:
            y = float(met[0])
        elif b >= br[-1]:
            y = float(met[-1])
        else:
            y = float(np.interp(b, br, met))

        # Khusus Accuracy, clamp ke [0, 1]
        if self.metric_col.lower() == "accuracy":
            y = max(0.0, min(1.0, y))
        return y
        
    def bitrate_bounds_for(self, k: Knob):
        key = (k.stream, k.model, k.QP, k.FPS, k.Resolution)
        if key not in self.curves:
            raise KeyError(f"No curve for {key}")
        br, _ = self.curves[key]
        return float(br[0]), float(br[-1])

class Profiler:
    def __init__(self, csv_path: str):
        import os
        ext = os.path.splitext(csv_path)[1].lower()

        # 1) Baca file (Excel atau CSV). CSV: auto-detect delimiter, strip BOM.
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

        # 2) Normalisasi nama kolom: trim spasi, samakan kapitalisasi
        #    (tetap pertahankan nama standar yang dipakai di script).
        df.columns = [c.strip() for c in df.columns]

        # 3) Pastikan nama kolom sesuai ekspektasi (alias ringan)
        #    Jika kamu pakai huruf kecil semua, sesuaikan mapping di sini.
        rename_map = {
            "stream": "stream",
            "model": "model",
            "qp": "QP",
            "fps": "FPS",
            "resolution": "Resolution",
            "accuracy": "Accuracy",
            "bitrate": "Bitrate",
            "latency": "Latency",
            "gpu_load": "GPU_load",
            "gpu": "GPU_load",          # alias umum
            "gpuload": "GPU_load",
        }
        # terapkan rename_map hanya bila key ada
        lower2orig = {c.lower(): c for c in df.columns}
        for low, std in rename_map.items():
            if low in lower2orig and std not in df.columns:
                df.rename(columns={lower2orig[low]: std}, inplace=True)

        # 4) Validasi kolom wajib
        required = ["stream","model","QP","FPS","Resolution","Accuracy","Bitrate","Latency","GPU_load"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

        # 5) Tipe data
        df["stream"] = df["stream"].astype(int)
        df["QP"] = df["QP"].astype(int)
        df["FPS"] = df["FPS"].astype(int)
        df["Bitrate"] = df["Bitrate"].astype(float)
        df["Resolution"] = df["Resolution"].astype(str)

        # 6) Simpan & bangun estimator
        self.df = df
        self.acc = CurveEstimator(self.df, "Accuracy")
        self.lat = CurveEstimator(self.df, "Latency")
        self.gpu = CurveEstimator(self.df, "GPU_load")


    def query(self, k: Knob, b: float):
        # try exact row first
        m = self.df
        sel=(m.stream==k.stream)&(m.model==k.model)&(m.QP==k.QP)&(m.FPS==k.FPS)&(m.Resolution==k.Resolution)&(np.isclose(m.Bitrate,b))
        if sel.any():
            r = m[sel].iloc[0]; return float(r.Accuracy), float(r.Latency), float(r.GPU_load)
        return self.acc.predict(k,b), self.lat.predict(k,b), self.gpu.predict(k,b)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--stream", type=int, required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--qp", type=int, required=True)
    ap.add_argument("--fps", type=int, required=True); ap.add_argument("--res", required=True)
    ap.add_argument("--bitrate", type=float, required=True)
    a=ap.parse_args()
    p=Profiler(a.csv); k=Knob(stream=a.stream, model=a.model, QP=a.qp, FPS=a.fps, Resolution=a.res)
    acc,lat,gpu = p.query(k,a.bitrate)
    print(json.dumps({"stream":k.stream,"model":k.model,"QP":k.QP,"FPS":k.FPS,"Resolution":k.Resolution,
                      "Bitrate":a.bitrate,"Accuracy":round(acc,6),"Latency":round(lat,6),
                      "GPU_load":round(gpu,6)}, indent=2))
if __name__=="__main__": main()
