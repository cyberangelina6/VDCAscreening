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


