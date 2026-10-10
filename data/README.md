# Included recordings

`raw/notr.csv`, `raw/sag.csv`, `raw/sol.csv`, and `raw/kirpma.csv` are the supplied continuous EOG recordings for neutral, right, left, and long blink respectively. Each row contains one numeric signal value; units and original sampling frequency were not recorded in the files. Replay mode assumes 330 Hz, as in the supplied code.

The data owner confirmed that these recordings belong to İlayda Sarıkoca and authorized public sharing. They do not contain participant names or identifiers. They remain physiological recordings and should be treated respectfully.

These four streams are examples for inspection and playback, **not the complete labeled event dataset used for the reported 95.9% test result**. Do not evaluate the model by treating a whole stream as one labeled event. The model expects centered event windows of about 1.1 seconds, prepared by the shared preprocessing function.

Training expects one event per CSV in `data/training/NOTR/<participant>/*.csv`, `SAG/<participant>/*.csv`, `SOL/<participant>/*.csv`, and `GOZ_KIRPMA/<participant>/*.csv`. That dataset is not included. Participant/session separation is needed for any claim about generalization to unseen people.
