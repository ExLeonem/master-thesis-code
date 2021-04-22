FROM jupyter/scipy-notebook:latest

# Update conda
RUN conda update -n base conda

# Install packages needed for moment propagation
RUN conda install -c conda-forge tqdm==4.59.0
RUN conda install -c anaconda tensorflow==2.3.0
RUN conda install -c anaconda tensorflow_probability==0.12.1
