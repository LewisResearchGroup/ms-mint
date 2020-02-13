# The browser based GUI
The app's frontend is build using [Plotly-Dash](https://plot.ly/dash/) and runs locally in a browser. Thought, the Python functions provided can be imported and used in any Python project independently.


### Load files
#### Mass spec files (mzML or mzXML)
Individual files can be added to an in worklist using the `ADD FILE(S)` button. If the checkbox `Add files from directory` is checked, all files from a directory and its subdirectories are imported that end on `mzXML` or `mzML` (not yet supported). The box is checked by default. Note that files are always added to the worklist. The worklist can be cleared with the `CLEAR FILES` button, which has no effect on the selected peaklists.

#### Select Peaklist file
A standard peaklist is provided. A user defined peaklist can be selected and used with the `SELECT PEAKLIST FILE(S)` button. Peaklists are explained in more detail [here](index.md#peaklists).

![Demo Image](./image/mint1.png "Demo image")


### Running Mint
The number of cores used for MINT can be selected with the `Select number of cores` slider. The maximum number shown here depends on the computer on which MINT is running. The `RUN` button starts mint and a progress bar monitors the progress and can be used to estimate the remaining time.

### Export
Mint results can be exported using the `export()` method. A filename has to be provided:

    mint.export('MINT-results.xlsx')

An Excel file with the results is exported. The file includes the data of all windows that were used as well as some metadata such as date and the current verion of `Mint`.

# Interactive elements

## Interative Results Table
The results

## Heatmap Tool
![Demo Image](./image/mint2.png "Demo image")


## Peak-Viewer
![Demo Image](./image/mint3.png "Demo image")


## 3D-Peak-Viewer
![Demo Image](./image/mint4.png "Demo image")

