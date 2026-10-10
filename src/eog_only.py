# -*- coding: utf-8 -*-
"""
GERCEK-ZAMAN EKRAN  --  EOG sinyalini alir ve ekrana buyuk harflerle
SAG / SOL / NOTR / UZUN KIRPMA yazar.

YENI: OLAY-MERKEZLI siniflandirma. Program "son 1 saniye"ye degil,
tamponda bir OLAY (belirgin sapma) arar, onu dip'e ORTALAYIP oyle
siniflandirir -- egitimdeki gibi. Sol artik dogru taninir.

IKI MOD:
  1) CANLI (ESP'den):     python gercek_zaman.py
  2) TEST (donanimsiz):   python gercek_zaman.py bir_dosya.csv
"""
import sys, time, threading, os
from pathlib import Path
from collections import deque, Counter
import tkinter as tk
import numpy as np
import joblib
from eog_ortak import hazirla, CLASSES

# ---------------- AYARLAR ----------------
SERIAL_PORT = os.environ.get("EOG_SERIAL_PORT", "COM16")         # CANLI modda kendi portunu yaz
BAUD_RATE   = 115200
MODEL_YOLU  = os.environ.get("EOG_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "model_eog.pkl"))
PENCERE_SN  = 1.1             # olay penceresi (egitim ~1.1 sn)
BUFFER_SN   = 2.5            # tamponda kac saniye tutulsun (olayi ortalamak icin)
GUNCELLE_MS = 150            # ekran kac ms'de bir guncellensin
THR_OLAY    = 350            # bu kadar sapma yoksa "olay yok" -> NOTR
CONF_ESIK   = 0.55           # bu olasiligin altinda oyu degistirme
OY_N        = 3              # son N tahmin (titremeyi onler)

RENK = {"notr": "#888780", "sag": "#185FA5", "sol": "#3B6D11",
        "kirpma": "#534AB7", "...": "#B4B2A9"}
ETIKET = {"notr": "NOTR", "sag": "SAG", "sol": "SOL", "kirpma": "UZUN KIRPMA"}

# ---------------- DURUM ----------------
model = joblib.load(MODEL_YOLU)
buffer = deque()
buffer_lock = threading.Lock()
oylar = deque(maxlen=OY_N)
fs = 200

# ---------------- VERI KAYNAKLARI ----------------
def seri_oku():
    global fs
    import serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2); ser.reset_input_buffer()
    t0 = time.time(); n = 0
    while time.time() - t0 < 2.0:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            try:
                with buffer_lock: buffer.append(float(line))
                n += 1
            except ValueError: pass
    fs = max(50, int(n / 2.0))
    print(f"Olculen ornekleme hizi ~{fs} Hz")
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            try:
                with buffer_lock:
                    buffer.append(float(line))
                    while len(buffer) > int(fs * BUFFER_SN) + 5:
                        buffer.popleft()
            except ValueError:
                pass

def csv_oynat(path):
    global fs
    veri = np.loadtxt(path)
    fs = 330
    print(f"TEST modu: {path} oynatiliyor ({len(veri)} ornek)")
    for v in veri:
        with buffer_lock:
            buffer.append(float(v))
            while len(buffer) > int(fs * BUFFER_SN) + 5:
                buffer.popleft()
        time.sleep(1.0 / fs)
    print("Oynatma bitti.")

# ---------------- TAHMIN (OLAY-MERKEZLI) ----------------
def tahmin_et():
    pencere = int(fs * PENCERE_SN)
    if len(buffer) < pencere + 10:
        return "..."
    with buffer_lock:
        sig = np.array(buffer, dtype=float)
    base = np.median(sig)
    dev = sig - base
    ti = int(np.argmax(np.abs(dev)))      # en belirgin sapma (olay merkezi)
    if abs(dev[ti]) < THR_OLAY:
        return "notr"                     # olay yok -> NOTR
    half = pencere // 2
    s = ti - half; e = ti + half
    if s < 0 or e > len(sig):
        return None                       # olay henuz ortalanamiyor -> bekle
    girdi = hazirla(sig[s:e])
    if girdi is None:
        return None
    olaslik = model.predict_proba(girdi.reshape(1, -1))[0]
    i = int(np.argmax(olaslik))
    if olaslik[i] < CONF_ESIK:
        return None
    return CLASSES[i]

# ---------------- EKRAN GUNCELLEME ----------------
def guncelle():
    ham = tahmin_et()
    if ham is not None:                   # None ise oyu degistirme
        oylar.append(ham)
    if oylar:
        sinif = Counter(oylar).most_common(1)[0][0]
    else:
        sinif = "..."
    yazi = "..." if sinif == "..." else ETIKET.get(sinif, sinif.upper())
    renk = RENK.get(sinif, "#000")
    label.config(text=yazi, fg=renk)
    kok.after(GUNCELLE_MS, guncelle)

# ---------------- BASLAT ----------------
kok = tk.Tk()
kok.title("EOG Canli Tahmin")
kok.geometry("700x400")
kok.configure(bg="white")
label = tk.Label(kok, text="...", font=("Arial", 80, "bold"), bg="white", fg="#888")
label.pack(expand=True)
tk.Label(kok, text="Her hareketi ~1 saniye net tut, sonra merkeze don",
         font=("Arial", 13), bg="white", fg="#888").pack(pady=10)

if len(sys.argv) > 1:
    threading.Thread(target=csv_oynat, args=(sys.argv[1],), daemon=True).start()
else:
    threading.Thread(target=seri_oku, daemon=True).start()

kok.after(1000, guncelle)
kok.mainloop()
