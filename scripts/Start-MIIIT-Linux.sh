PORT=9831
source /home/`whoami`/.bashrc
source activate miiit
cd /home/swacker/workspace/uofc/code/metabolomics/MIIIT
xdg-open "http://localhost:${PORT}/apps/notebooks/Metabolomics_Interactive_Intensity_Integration_Tool.ipynb?appmode_scroll=0"
jupyter notebook --no-browser --port=${PORT}

