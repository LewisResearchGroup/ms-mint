# Peak detection

Peak detection is still an experimental feature. OpenMS's algorithm for peak detection from metabolomics data 'FeatureFindingMetabo' is employed. Peak detection happens in two steps. First, each selected file is scanned for potential peaks. Then similar peaks across multiple files are merged. Peak alignment for retention time drift correction is omitted. The feature maps that are internally stored in OpenMS format are then transformed to MINT peaklist format. The detected features often come from contaminations and additional scoring step is recommended to filter out low quality peak definitions. The controllable parameters for the detection algorithm are the minimum OpenMS peak quality and the maximum number of peaks per file. 

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

