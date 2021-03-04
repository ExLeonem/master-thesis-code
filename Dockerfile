FROM jupyter/scipy-notebook:latest

# Update conda
RUN conda update -n base conda

# Deep learning framework
# RUN conda install -c anaconda tensorflow-gpu 
# RUN conda install -c conda-forge tensorflow-probability 
RUN conda install pytorch cudatoolkit=10.1 -c pytorch

# --------- Packages Computer Vision --------
RUN conda install -c menpo opencv

# RUN conda install -c menpo cyvlfeat
# RUN conda install -c conda-forge vlfeat
# RUN conda install -c conda-forge tesseract
# RUN conda install -c conda-forge pytesseract

# --------- Packages for NLP
# ---- Spacy
# RUN conda install -c conda-forge spacy
# RUN python -m spacy download de
# RUN python -m spacy download en
# RUN python -m spacy download en_core_web_sm

# ---- NLTK
# RUN conda install -c conda-forge nltk