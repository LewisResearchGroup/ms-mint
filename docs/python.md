# Python API

`ms-mint` backend can be imported as a python object and used in a python script or interactively in a Jupyter notebook environment. A typical workflow using the python API is described in the following:

    from ms_mint import Mint
    from glob import glob


Fist, the Mint class has to be instantiated:

    mint = Mint(verbose=False)

### Load files
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

### Running Mint
Then mint can be executed calling the `run()` method:

    mint.run()

    > Run MINT
    > Total runtime: 6.18s
    > Runtime per file: 3.09s
    > Runtime per peak (79): 0.04s


### Results
The result will be stored in the `results` and the `crosstab` attributes as `pandas.DataFrames()`. Where `mint.results` contains all results:

    print(mint.results)

<pre>
peak_label    mz_mean  mz_width  rt_min  rt_max  intensity_threshold  peaklist  peak_area  ms_file ms_path  file_size  intensity_sum
1  151.06050  5    5.07    5.09  0  ./data/peaklist_v0.csv  2.879748e+03  ./data/test.mzXML  ./data  14.201964   5.607296e+10
2  216.05040  5    3.98    4.39  0 ./data/peaklist_v0.csv  4.892307e+05  ./data/test.mzXML  ./data  14.201964   5.607296e+10
3  115.00320  5    3.45    4.39  0  ./data/peaklist_v0.csv  3.916772e+07  ./data/test.mzXML  ./data  14.201964   5.607296e+10
4  273.00061  5    1.10    2.22  0  ./data/peaklist_v0.csv  6.862484e+06  ./data/test.mzXML  ./data  14.201964   5.607296e+10
1  151.06050  5    5.07    5.09  0  ./data/peaklist_v0.csv  2.879748e+03  ./data/test.mzXML  ./data  14.201964   5.607296e+10
2  216.05040  5    3.98    4.39  0  ./data/peaklist_v0.csv  4.892307e+05  ./data/test.mzXML  ./data  14.201964   5.607296e+10
3  115.00320  5    3.45    4.39  0  ./data/peaklist_v0.csv  3.916772e+07  ./data/test.mzXML  ./data  14.201964   5.607296e+10
4  273.00061  5    1.10    2.22  0  ./data/peaklist_v0.csv  6.862484e+06  ./data/test.mzXML  ./data  14.201964   5.607296e+10
</pre>

and `crosstab()` can shows a compressed form of the data only containing one property e.g. the extracted `peak_area`:

<pre>
...
</pre>


### Peak-shapes
The last property is `mint.rt_projections` which stores a dictionary of dictionaries with peak shapes:

<pre>
{'1': {'./data/test.mzXML': retentionTime
  5.079267    2879.747559
  dtype: float32},
 '2': {'./data/test.mzXML': retentionTime
  3.986050    15166.202148
  3.996917    14039.182617
  4.007817    15455.113281
  4.018700    16612.851562
  4.029633    22065.619141
  4.040633    26693.970703
  4.051533    22569.896484
  4.062450    32379.552734
  4.073567    27225.439453
  4.084683    22142.037109
  4.095867    22974.357422
  4.106900    23733.207031
  4.117917    25081.419922
  4.128983    17945.343750
  4.140200    14623.268555
  4.151250    16119.997070
  4.162317    15771.708008
  4.173467    11171.838867
  4.184517    12554.623047
  ...
</pre>

#### Plotting shapes
The peak shapes can be plotted with the same function that is used by the GUI's:

    from ms_mint.plotly_tools import plot_rt_projections
    plot_rt_projections(mint)

### Export results
Mint results can be exported using the `export()` method. A filename has to be provided:

    mint.export('MINT-results.xlsx')
