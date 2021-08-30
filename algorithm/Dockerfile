FROM python:3

ADD module.py /
WORKDIR /algorithm
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip install pathlib==1.0.1
RUN pip install numpy==1.20.3
RUN pip install matplotlib==3.4.2
RUN pip install qiskit==0.26.2
RUN pip install discopy==0.3.5
RUN pip install pytket==0.11.0
RUN pip install pytket-qiskit==0.13.0
