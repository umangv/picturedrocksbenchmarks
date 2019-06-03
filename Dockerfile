FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install -y git make
RUN conda install python=3.7.* h5py
RUN mkdir -p /benchmarks/
COPY . /benchmarks/ 
WORKDIR /benchmarks/
RUN pip install -r requirements.txt
