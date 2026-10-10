# -*- coding: utf-8 -*-
"""
GERCEK-ZAMAN + MOTOR KONTROL
EOG sinyalini ESP'den alir, jesti tanir ve MOTOR Arduino'suna komut yollar.

  EOG ESP (SERIAL_PORT)  -> PC siniflandirir -> MOTOR Arduino (MOTOR_PORT)

Jest -> komut:
  uzun kirpma -> 'f' (ileri/dur toggle)
  saga bak    -> 'r' (90 derece sag)
  sola bak    -> 'l' (90 derece sol)
  notr        -> komut yok

GUVENLIK: bir jest = bir komut. Komut gonderilince, tekrar NOTR'e
donulene kadar yeni komut gonderilmez.

MOD:
  CANLI:  python gercek_zaman.py
  TEST :  python gercek_zaman.py dosya.csv   (motor olmadan ekran testi)
"""
import sys, time, threading, os
from pathlib import Path
from collections import deque, Counter
import tkinter as tk
import numpy as np
import joblib
from eog_ortak import hazirla, CLASSES

# ---------------- AYARLAR ----------------
SERIAL_PORT = os.environ.get("EOG_SERIAL_PORT", "COM16")        # EOG ESP portu
MOTOR_PORT  = os.environ.get("MOTOR_SERIAL_PORT", "COM17")         # MOTOR Arduino portu  <-- Aygit Yoneticisi'nden bak, duzelt
BAUD_RATE   = 115200         # ESP
MOTOR_BAUD  = 9600           # motor Arduino (kodunda Serial.begin(9600))
MODEL_YOLU  = os.environ.get("EOG_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models" / "model_eog.pkl"))
PENCERE_SN  = 1.1
BUFFER_SN   = 2.5
GUNCELLE_MS = 150
THR_OLAY    = 350
CONF_ESIK   = 0.55
OY_N        = 3
OTOMATIK_START = os.environ.get("EOG_AUTO_START", "0") == "1"        # baglaninca motor Arduino'ya 'start' gonderilsin mi

KOMUT_MAP = {"sag": "r", "sol": "l", "kirpma": "f"}
KOMUT_YAZI = {"r": "-> SAGA DON", "l": "-> SOLA DON", "f": "-> ILERI / DUR"}

RENK = {"notr": "#888780", "sag": "#185FA5", "sol": "#3B6D11",
        "kirpma": "#534AB7", "...": "#B4B2A9"}
ETIKET = {"notr": "NOTR", "sag": "SAG", "sol": "SOL", "kirpma": "UZUN KIRPMA"}

# ---------------- DURUM ----------------
model = joblib.load(MODEL_YOLU)
buffer = deque()
buffer_lock = threading.Lock()
oylar = deque(maxlen=OY_N)
fs = 200
motor_ser = None
beklemede = True             # NOTR'e donuldu mu (yeni komut alinabilir mi)
son_komut_yazi = ""

# ---------------- MOTOR ----------------
def motor_baglan():
    global motor_ser
    try:
        import serial
        motor_ser = serial.Serial(MOTOR_PORT, MOTOR_BAUD, timeout=0.1)
        time.sleep(2)                       # Arduino reset suresi
        if OTOMATIK_START:
            motor_ser.write(b"start\n")
        print("Motor Arduino baglandi" + (" + START gonderildi" if OTOMATIK_START else ""))
    except Exception as e:
        motor_ser = None
        print("Motor baglanamadi (sadece ekran modu):", e)

def komut_gonder(k):
    global son_komut_yazi
    if motor_ser is not None:
        try: motor_ser.write((k + "\n").encode())
        except Exception as e: print("Komut gonderilemedi:", e)
    son_komut_yazi = KOMUT_YAZI.get(k, k)
    print("KOMUT GONDERILDI:", k)

# ---------------- EOG OKUMA ----------------
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
    veri = np.loadtxt(path); fs = 330
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
    if len(buffer) < pencere + 10: return "..."
    with buffer_lock:
        sig = np.array(buffer, dtype=float)
    base = np.median(sig); dev = sig - base
    ti = int(np.argmax(np.abs(dev)))
    if abs(dev[ti]) < THR_OLAY: return "notr"
    half = pencere // 2; s = ti - half; e = ti + half
    if s < 0 or e > len(sig): return None
    girdi = hazirla(sig[s:e])
    if girdi is None: return None
    olaslik = model.predict_proba(girdi.reshape(1, -1))[0]
    i = int(np.argmax(olaslik))
    if olaslik[i] < CONF_ESIK: return None
    return CLASSES[i]

# ---------------- EKRAN + KOMUT ----------------
def guncelle():
    global beklemede
    ham = tahmin_et()
    if ham is not None: oylar.append(ham)
    sinif = Counter(oylar).most_common(1)[0][0] if oylar else "..."

    # --- KOMUT MANTIGI: bir jest = bir komut, arada NOTR sart ---
    if sinif in ("notr", "..."):
        beklemede = True
    elif beklemede and sinif in KOMUT_MAP:
        komut_gonder(KOMUT_MAP[sinif])
        beklemede = False

    yazi = "..." if sinif == "..." else ETIKET.get(sinif, sinif.upper())
    label.config(text=yazi, fg=RENK.get(sinif, "#000"))
    komut_label.config(text=son_komut_yazi)
    kok.after(GUNCELLE_MS, guncelle)

# ---------------- BASLAT ----------------
kok = tk.Tk()
kok.title("EOG -> Motor Kontrol")
kok.geometry("700x440"); kok.configure(bg="white")
label = tk.Label(kok, text="...", font=("Arial", 80, "bold"), bg="white", fg="#888")
label.pack(expand=True)
komut_label = tk.Label(kok, text="", font=("Arial", 22, "bold"), bg="white", fg="#534AB7")
komut_label.pack()
tk.Label(kok, text="Her hareketi net yap, sonra NOTR'e don (arada notr sart)",
         font=("Arial", 12), bg="white", fg="#888").pack(pady=8)

if len(sys.argv) > 1:                       # TEST modu (CSV) - motor yok
    threading.Thread(target=csv_oynat, args=(sys.argv[1],), daemon=True).start()
else:                                        # CANLI mod - EOG + motor
    threading.Thread(target=seri_oku, daemon=True).start()
    if os.environ.get("EOG_ENABLE_MOTOR", "0") == "1":
        threading.Thread(target=motor_baglan, daemon=True).start()
    else:
        print("Motor transmission disabled. Set EOG_ENABLE_MOTOR=1 to enable.")

kok.after(1500, guncelle)
kok.mainloop()
