# QEL_Merck_Challenge
QNLP for adverse events detection
sudo docker run --name qtex -d -p 4200:80 qtex
Run the system using : http://localhost:4200/
HTTP Port 80
The notebook to run is named "ADE_detection.ipynb"
The training dataset and testing dataset are in the folder "datasets"
The folder "experiment_results" contains results of 15 sample runs, including parameters and training errors at each iteration as well as testing errors for every 10 iterations. The averaged training error and testing error trajectory is also presented as .png there.