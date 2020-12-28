# Peaklists
A peaklist contains the definitions of peaks to be extracted. It is the protocol that captures how data is going to be extracted from the MS-files. It is provided as `csv` (comma separated values) or `xlsx` (Microsoft Excel) file. Each row defines a peak-window. If a multi-sheet xlsx file can be used the peaklist should be the first sheet.

A window is defined by five properties that need to be present as headers in the peaklist file:

- **peak_label** : A __unique__ identifier such as the biomarker name or ID. Even if multiple peaklist files are used, the label have to be unique across all the files.
- **mz_mean** : The target mass (m/z-value) in [Da].
- **mz_width** : The width of the peak in the m/z-dimension in units of ppm. The window will be *mz_mean* +/- (mz_width * mz_mean * 1e-6). Usually, a values between 5 and 10 are used.
- **rt** : Estimated retention time in [min].
- **rt_min** : The start of the retention time for each peak in [min].
- **rt_max** : The end of the retention time for each peak in [min].
- **intensity_threshold** : A threshold that is applied to filter noise for each window individually. Can be set to 0 or any positive value.

#### Example file
**peaklist.csv:**
```text
peak_label,mz_mean,mz_width,rt_min,rt_max,intensity_threshold
Biomarker-A,151.0605,10,4.65,5.2,0
Biomarker-B,151.02585,10,4.18,4.53,0
```

A template can be created using the [GUI](gui.md).
