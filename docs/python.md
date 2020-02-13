# Python Integration

`ms-mint` backend can be imported as a python object and used in a python script or interactively in a Jupyter notebook environment. A typical workflow using the python API is described in the following:

    from ms_mint import Mint
    from glob import glob


Fist, the Mint class has to be instantiated:

    mint = Mint(verbose=False)

One ore more peaklist files as well as mass-spec files have to be assigned to `mint.peaklist_files` and `mint.files` accordingly:

    mint.peaklist_files = ['path_to/peaklist-file.csv']
    mint.files = glob('path_to/ms-files/*/**.mzML', recursive=True)

Calling the `peaklist()` method displays the imported and concatenated peaklists:

    print(mint.peaklist)

<pre>
  peak_label    mz_mean  mz_width  rt_min  rt_max  intensity_threshold                peaklist
0          1  151.06050         5    5.07    5.09                    0  ./data/peaklist_v0.csv
1          2  216.05040         5    3.98    4.39                    0  ./data/peaklist_v0.csv
2          3  115.00320         5    3.45    4.39                    0  ./data/peaklist_v0.csv
3          4  273.00061         5    1.10    2.22                    0  ./data/peaklist_v0.csv
</pre>

Then mint can be executed calling the `run()` method:

    mint.run()

    > Run MINT
    > Total runtime: 6.18s
    > Runtime per file: 3.09s
    > Runtime per peak (79): 0.04s



The result will be stored in the `results` and the `crosstab` attributes:

    print(mint.results)

<pre>
  peak_label    mz_mean  mz_width  rt_min  rt_max  intensity_threshold                peaklist     peak_area            ms_file ms_path  file_size  intensity_sum
0          1  151.06050         5    5.07    5.09                    0  ./data/peaklist_v0.csv  2.879748e+03  ./data/test.mzXML  ./data  14.201964   5.607296e+10
1          2  216.05040         5    3.98    4.39                    0  ./data/peaklist_v0.csv  4.892307e+05  ./data/test.mzXML  ./data  14.201964   5.607296e+10
2          3  115.00320         5    3.45    4.39                    0  ./data/peaklist_v0.csv  3.916772e+07  ./data/test.mzXML  ./data  14.201964   5.607296e+10
3          4  273.00061         5    1.10    2.22                    0  ./data/peaklist_v0.csv  6.862484e+06  ./data/test.mzXML  ./data  14.201964   5.607296e+10
0          1  151.06050         5    5.07    5.09                    0  ./data/peaklist_v0.csv  2.879748e+03  ./data/test.mzXML  ./data  14.201964   5.607296e+10
1          2  216.05040         5    3.98    4.39                    0  ./data/peaklist_v0.csv  4.892307e+05  ./data/test.mzXML  ./data  14.201964   5.607296e+10
2          3  115.00320         5    3.45    4.39                    0  ./data/peaklist_v0.csv  3.916772e+07  ./data/test.mzXML  ./data  14.201964   5.607296e+10
3          4  273.00061         5    1.10    2.22                    0  ./data/peaklist_v0.csv  6.862484e+06  ./data/test.mzXML  ./data  14.201964   5.607296e+10
</pre>

