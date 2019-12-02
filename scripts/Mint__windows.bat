
if exist C:\Users\%USERNAME%\AppData\Local\Continuum\miniconda3\Scripts\activate (
    call  C:\Users\%USERNAME%\AppData\Local\Continuum\miniconda3\Scripts\activate ms-mint
) else if exist C:\Users\%USERNAME%\Miniconda3\Scripts\activate (
    call C:\Users\%USERNAME%\Miniconda3\Scripts\activate ms-mint
) else (
    echo Could not find anaconda's script 'activate'.
    echo Please, make sure it is in the PATH.
    call activate mint
)

python Mint.py