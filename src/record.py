# -*- coding: utf-8 -*-
"""
KAYIT  --  ESP'den gelen sinyali bir CSV dosyasina kaydeder.

Kullanim:
    py kayit.py sag.csv 60      ->  60 saniye kayit, sag.csv'ye yazar
    py kayit.py notr.csv 60
    py kayit.py sol.csv 60
    py kayit.py kirpma.csv 60

ONEMLI: Arduino Serial Monitor KAPALI olmali (port tek programda acilir).
"""
import sys, time, serial, os

PORT = os.environ.get("EOG_SERIAL_PORT", "COM16")          # kendi COM portun
BAUD = 115200

def main():
    if len(sys.argv) < 2:
        print("Kullanim: py kayit.py CIKIS.csv [saniye]")
        return
    dosya = sys.argv[1]
    sure = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2); ser.reset_input_buffer()

    print(f"\n'{dosya}' icin {sure:.0f} saniye kayit alinacak.")
    print("Hazir ol, geri sayim:")
    for k in (3, 2, 1):
        print(f"   {k}...")
        time.sleep(1)
    print(">>> BASLA! Hareketi tekrar tekrar yap. <<<\n")

    vals = []
    t0 = time.time()
    son_yazim = 0
    while time.time() - t0 < sure:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            try:
                vals.append(float(line))
            except ValueError:
                pass
        gecen = int(time.time() - t0)
        if gecen != son_yazim:                  # her saniye kalan sureyi yaz
            print(f"   kalan: {int(sure)-gecen} sn   (toplanan: {len(vals)})", end="\r")
            son_yazim = gecen
    ser.close()

    with open(dosya, "w") as f:
        for v in vals:
            f.write(f"{v}\n")
    print(f"\n\nBitti. {len(vals)} ornek kaydedildi -> {dosya}")

if __name__ == "__main__":
    main()
