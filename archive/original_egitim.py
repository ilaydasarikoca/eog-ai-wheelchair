# -*- coding: utf-8 -*-
"""
EGITIM  --  veri klasorunden modeli egitir ve 'model_eog.pkl' olarak kaydeder.

Klasor yapisi:
    VERI/
      NOTR/<kisi>/*.csv
      SAG/<kisi>/*.csv
      SOL/<kisi>/*.csv
      GOZ_KIRPMA/<kisi>/*.csv

Kullanim:  python egitim.py VERI
"""
import os, sys, glob, numpy as np, joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from eog_ortak import hazirla, CLASSES

KLASOR_ESLESME = {"NOTR": "notr", "SAG": "sag", "SOL": "sol", "GOZ_KIRPMA": "kirpma"}

def veri_yukle(base):
    X, y = [], []
    for klasor, sinif in KLASOR_ESLESME.items():
        n = 0
        for f in glob.glob(os.path.join(base, klasor, "**", "*.csv"), recursive=True):
            try:
                sig = np.loadtxt(f)
            except Exception:
                continue                       # bos / bozuk dosya -> atla
            v = hazirla(sig)
            if v is None:
                continue
            X.append(v); y.append(CLASSES.index(sinif)); n += 1
        print(f"  {sinif:7s}: {n} gecerli dosya")
    return np.array(X), np.array(y)

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "VERI"
    print(f"Veri yukleniyor: {base}")
    X, y = veri_yukle(base)
    if len(X) == 0:
        print("HATA: hic veri bulunamadi. Klasor yolunu kontrol et.")
        return
    print(f"Toplam {len(X)} ornek.\n")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)
    clf.fit(Xtr, ytr)

    acc = accuracy_score(yte, clf.predict(Xte))
    print(f">>> TEST DOGRULUGU: %{acc*100:.1f}\n")
    cm = confusion_matrix(yte, clf.predict(Xte))
    print("Confusion (satir=gercek, sutun=tahmin)")
    print("        " + " ".join(f"{c:>8s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>8s} " + " ".join(f"{x:8d}" for x in cm[i]))

    joblib.dump(clf, "model_eog.pkl")
    print("\nModel kaydedildi -> model_eog.pkl")

if __name__ == "__main__":
    main()
