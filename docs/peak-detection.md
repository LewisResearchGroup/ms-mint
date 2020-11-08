# Peak detection

Peak detection is still an experimental feature. Internally, OpenMS's algorithm for peak detection from metabolomics data 'FeatureFindingMetabo' is used to detect peaks in two steps. First, each MS file is scanned for potential peaks with the OpenMS algorithm. Then similar peaks across multiple files are merged. Peak alignment for retention time drift correction is omitted. The feature maps that are internally stored in OpenMS format (featureXML) are then transformed to table format that is used by MINT. However, the detected features often come from contaminations. Additional scoring and pre-filtering is recommended to filter out low quality peak definitions which is currently investigated. The controllable parameters for the detection algorithm are the minimum OpenMS peak quality and the maximum number of peaks per file.

In the Jupter Notebook feature detection can be done as demonstrated in the following:

    %pylab inline

    from ms_mint.notebook import Mint
    from glob import glob

    mint = Mint()
    mint.show()

![](image/jupyter.png)


    mint.ms_files = [list-of-filenames]
    
    mint.detect_peaks(min_quality=1e-3, condensed=True, 
                      max_delta_mz_ppm=10, max_delta_rt=0.1)

