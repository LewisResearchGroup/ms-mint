
if exist C:\Users\%USERNAME%\AppData\Local\Continuum\miniconda3\Scripts\activate (
    call  C:\Users\%USERNAME%\AppData\Local\Continuum\miniconda3\Scripts\activate miiit
) else (
    echo Could not find anaconda's script 'activate'.
    echo Please, make sure it is in the PATH.
    call activate miiit
)

start "" http://localhost:8888/apps/notebooks/Metabolomics_Interactive_Intensity_Integration_Tool.ipynb?appmode_scroll=0
jupyter notebook --no-browser &
