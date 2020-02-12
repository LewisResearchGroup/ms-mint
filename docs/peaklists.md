# Peaklists
A peaklist is the protocol that captures how data is going to be extracted from the individual MS-files. It is provided as `csv` file and essentially contains the definitions of peaks to be extracted. A single peak is defined by five properties that need to be present as headers in the `csv` file which will be explained in the following:


#### peak_label
A **unique** identifier such as the biomarker name or ID. Even if multiple peaklist files are used, the label have to be unique across all the files.


#### mz_mean
The target mass (m/z-value) in [Da].


#### mz_width
The width of the peak in the m/z-dimension in units of ppm. The window will be *mz_mean* +/- (*mz_width*mz_mean*1e-6*). Usually, a values between 5 and 10 are used.


#### rt_min
The start of the retention time for each peak in [min].


#### rt_max
The end of the retention time for each peak in [min].


#### intensity_threshold
A threshold that is applied to filter noise for each window individually. Can be set to 0 or any positive value.


## Example Peaklist

`peaklist.csv`
```text
peak_label,mz_mean,mz_width,rt_min,rt_max,intensity_threshold
Biomarker-1,151.0605,10,4.65,5.2,0
Biomarker-2,151.02585,10,4.18,4.53,0
```
