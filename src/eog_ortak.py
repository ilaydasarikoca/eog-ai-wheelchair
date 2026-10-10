# -*- coding: utf-8 -*-
"""
ORTAK ON ISLEME  --  hem egitim hem gercek-zaman bu dosyayi kullanir.
Boylece ikisi BIREBIR ayni islemi yapar (en sik hata kaynagi budur).
Tum dosyalari AYNI klasorde tut.

NOT: Bu surum, sag/sol'u ayirmak icin DONUS-SEKLI ozelliklerini de ekler
(sol'un sonundaki keskin dip vs sag'in yumusak donusu). Cikti 209 boyut.
"""
import numpy as np
from scipy.signal import resample

L = 200                                  # her sinyal bu kadar noktaya getirilir
CLASSES = ["notr", "sag", "sol", "kirpma"]

def _smooth(w, k=9):
    return np.convolve(w, np.ones(k) / k, mode='same')

def _features(s):
    return np.array([
        s.max(), s.min(), s.std(),
        np.sum(s ** 2) / len(s),               # guc
        ((s[:-1] * s[1:]) < 0).sum(),          # zero-crossing
    ])

def _donus(s):
    """DONUS-SEKLI ozellikleri -- sag ile sol'u ayiran kisim.
    Ana dip sonrasi: ikincil dip derinligi, belirginligi, sertlik, puruzluluk."""
    sm = _smooth(s, 9)
    tro = int(np.argmin(sm))                   # ana dip
    ret = sm[tro:]                             # donus parcasi
    if len(ret) < 30:
        return np.array([0.0, 0.0, 0.0, 0.0])
    half = max(2, len(ret) // 2)
    pk = int(np.argmax(ret[:half]))            # dip sonrasi toparlanma tepesi
    after = ret[pk:]
    sec = float(after.min()) if len(after) else 0.0   # ikincil dip (sol'da derin)
    prom = float(ret[pk] - sec)                # ikincil dipin belirginligi
    d = np.diff(ret)
    return np.array([sec, prom, float(np.max(np.abs(d))), float(np.std(d))])

def hazirla(sinyal):
    """Ham EOG sinyali -> 209 boyutlu model girisi. Hataliysa None doner."""
    s = np.asarray(sinyal, dtype=float).ravel()
    if s.size < 10 or not np.isfinite(s).all():
        return None
    s = resample(s, L)                         # ornekleme hizindan bagimsiz
    s = s - np.mean(s)                         # baseline cikar
    s = s / (np.std(s) + 1e-7)                 # z-score normalize
    return np.concatenate([s, _features(s), _donus(s)])
