# Quickstart

Here’s a minimal working example to get started with `ms-mint` in just a few steps:

### 1. Install the package

```bash
pip install ms-mint
```

---

### 2. Import the library

If you're using a **script**:

```python
from ms_mint.Mint import Mint
mint = Mint()
```

If you're using a **Jupyter notebook**:

```python
%pylab inline
from ms_mint.notebook import Mint
mint = Mint()
```

---

### 3. Load your mass spectrometry data files

You can load individual files:

```python
mint.ms_files = [
    './input/sample1.mzML',
    './input/sample2.mzXML',
]
```

Or use wildcards to load multiple files:

```python
mint.load_files('./input/*.mzML')
```

---

### 4. Load your target list

From a CSV file:

```python
mint.load_targets('targets.csv')
```

Or directly from a pandas DataFrame:

```python
import pandas as pd

targets = pd.read_csv('targets.csv')
mint.targets = targets
```

---

### 5. Run the analysis

```python
mint.run()
```

If you're working with **thousands of files**, save results directly to a file to save memory:

```python
mint.run(fn='results.csv')
```

---

### 6. View results

```python
mint.results
```

---

### Optional: Optimize retention time ranges

For better peak detection, especially if your RT values are estimates:

```python
mint.opt.rt_min_max(
    peak_labels=['Xanthine', 'Succinate'],
    plot=True
)
```

---

You’re now ready to process large-scale targeted metabolomics datasets with `ms-mint`!

Continue with [visualization](user_guide/visualization.md) or the structure of [target lists](user_guide/target_lists.md).
