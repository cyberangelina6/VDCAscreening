# Vocal Digital Cognitive Assessment Screening

Plain HTML/CSS/JS. The optional Python dataset tooling uses `ucimlrepo`.

## Run it

Open the HTML entry point in a browser.

## Python dataset tooling

Install the Python dependency with:

```bash
python -m pip install -r requirements.txt
```

Then fetch the Parkinsons Telemonitoring dataset:

```python
from ucimlrepo import fetch_ucirepo

parkinsons_telemonitoring = fetch_ucirepo(id=189)
X = parkinsons_telemonitoring.data.features
y = parkinsons_telemonitoring.data.targets

print(parkinsons_telemonitoring.metadata)
print(parkinsons_telemonitoring.variables)
```

## Extract features from recordings

The UCI Parkinson's CSV contains precomputed jitter, shimmer, fundamental
frequency, and HNR values, but it does not include the original audio. MFCCs
therefore cannot be recovered from that CSV. Use the supplied extractor with
the voice recordings associated with the observations:

```bash
python extract_acoustic_features.py path/to/recordings data/acoustic_features.csv
```

It recursively processes WAV, FLAC, OGG, MP3, M4A, and AIFF files. The output
contains local jitter and shimmer, HNR in dB, mean/median/standard deviation/
minimum/maximum F0 in Hz, voiced-frame fraction, and mean and standard
deviation for 13 MFCC coefficients. Jitter and shimmer are Praat ratios, so a
value of `0.01` corresponds to 1%.


