# Sources and release changes

Source archive: `EOG_son (2).zip`.

* EOG-only source: `en_iyi_model_sadece_eog_sonn`.
* Motor-linked source: `eog ve sürüş son`.
* Shortened-timing experiments and earlier variants are not presented as the final version.
* The supplied model is preserved byte-for-byte. Original source checksums are in `source_checksums.json`.

Release changes:

1. Deduplicated shared preprocessing, model, and example recordings.
2. Renamed executable scripts and made model paths relative to the repository, so launch location does not break model loading.
3. Added environment-based serial-port configuration; disabled motor transmission and automatic `start` by default. The original code enabled automatic start.
4. Added locks around shared-buffer snapshots to avoid concurrent-deque iteration failures. Event detection and class/command mapping remain unchanged.
5. Rejected non-finite signals in preprocessing.
6. Added a trainer matching the saved `StandardScaler + (96, 48) MLP` architecture, leaving the original `(64, 32)` trainer in `archive/original_egitim.py` as a reference.
7. Added a headless model check, installation instructions, tests and evaluation evidence.

The interim report is retained locally and not published: it contains student identifiers and describes an earlier, partly inconsistent architecture (including Keras and a seven-feature network). The final presentation is included as historical project evidence; its hardware descriptions, command mapping and safety claims are not all implemented by the Python files. For runtime behavior, follow the source code and README.

No reuse license has been selected on behalf of the team. Public visibility alone does not grant an open-source license.
