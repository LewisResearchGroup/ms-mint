# Target lists
A target list contains the definitions of peaks to be extracted in terms of retention time and mz value. The important parameters for MINT are `rt_min` and `rt_max`. The `rt` value is only used as an estimate and used for comparison. You should know, from former identification runs, at what retention time to expect a certain metabolite. This is what `rt` is for. For the final extraction process however `rt_min` and `rt_max` are used. Before you process the MS files, you should check that all targts have `rt_min` and `rt_max` properly set. 


### Target list format

The target list is the determining protocol for the data processing step. You can reproduce all results using this list as input. A target list can be provided as `csv` (comma separated values) or `xlsx` (Microsoft Excel) file. 

> If the preaklist is provided as multi-sheet xlsx file the target list should be the first sheet.

The input files contains a number of columns headers in the target list should contain:

- **peak_label** : A __unique__ identifier such as the biomarker name or ID. Even if multiple peaklist files are used, the label have to be unique across all the files.
- **mz_mean** : The target mass (m/z-value) in [Da].
- **mz_width** : The width of the peak in the m/z-dimension in units of ppm. The window will be *mz_mean* +/- (mz_width * mz_mean * 1e-6). Usually, a values between 5 and 10 are used.
- **rt** : Estimated retention time in [min] (optional, see above).
- **rt_min** : The start of the retention time for each peak in [min].
- **rt_max** : The end of the retention time for each peak in [min].
- **intensity_threshold** : A threshold that is applied to filter noise for each window individually. Can be set to 0 or any positive value.

#### Example file
**target.csv:**
```text
peak_label,mz_mean,mz_width,rt_min,rt_max,intensity_threshold
Biomarker-A,151.0605,10,4.65,5.2,0
Biomarker-B,151.02585,10,4.18,4.53,0
```


A template can be created using the [GUI](gui.md).
