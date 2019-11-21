# Peaklists
A peaklist is the protocol that captures how data is going to be extracted from the individual MS-files. It is provided as `csv` file and essentially contains the definitions of peaks to be extracted. A single peak is defined by five properties that need to be present as headers in the `csv` file.


#### peakLabel
A unique identifier such as the biomarker name or ID


#### peakMz
The target mass (m/z-value)


#### peakMzWidth[ppm]
The width of the peak in the m/z-dimension in units of ppm (peakMz/1e6)


#### rtmin
The start of the retention time for each peak in seconds.


#### rtmax
The end of the retention time for each peak in seconds.


## Example Peaklist

`peaklist.csv`
```text
peakLabel,peakMz,peakMzWidth[ppm],rtmin,rtmax
Arabitol,151.0605,10,4.65,5.2
Xanthine,151.02585,10,4.18,4.53
```
