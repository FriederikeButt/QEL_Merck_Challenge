# QEL_Merck_Challenge
QNLP for adverse events detection
sudo docker run --name qtex -d -p 4200:80 qtex
Run the system using : http://localhost:4200/
HTTP Port 80
The notebook to run is named "ADE_detection.ipynb"
The training dataset and testing dataset are in the folder "datasets"
The folder "experiment_results" contains results of 15 sample runs, including parameters and training errors at each iteration as well as testing errors for every 10 iterations. The averaged training error and testing error trajectory is also presented as .png there.


Step 1: Take clone: git clone https://github.com/FriederikeButt/QEL_Merck_Challenge.git

Step 2: Open the project in IDE (Recommended Visual Studio code)

Step 3: For running Dashboard(https://qtex.trudawnsolutions.com/)

cd qtex

Step 4: Build your docker image (If using visual studio right click command pallete docker image and select qelmerckchallenge:latest) Not: You must have docker already installed) To check images run "docker images" command in your terminal If somethng is not working use sudo

Step 5: For running Python code Docker that is algorithm
cd algorithm
=======
Step 4: Build your docker image
(If using visual studio right click command pallete docker image and select qelmerckchallenge:latest)
Not: You must have docker already installed)
To check images run "docker images" command in your terminal
If somethng is not working use sudo


Step 4: For running Python code
>>>>>>> 2e1cc4c91d8a69db0daf1e130fbb0d80f10c6f09
