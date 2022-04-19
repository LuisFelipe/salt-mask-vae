FROM tensorflow/tensorflow:1.14.0-gpu-py3

LABEL project.name="MASK-VAE: Models and Architectures for variational autoencorder of masks."\
      image.content="Experimetal environment."

RUN apt-get update &&\
    pip install pandas matplotlib jupyterlab ipywidgets &&\
    pip install pillow scikit-image seaborn "h5py" numpy tqdm scikit-learn pyyaml blinker learn-flow &&\
    pip install toposort invoke &&\
    mkdir -p /salt-mask-vae

WORKDIR /salt-mask-vae/src

ENTRYPOINT ["invoke"]
