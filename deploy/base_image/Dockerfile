FROM continuumio/miniconda

COPY env.yml .
RUN \
   conda env update -n root -f env.yml \
&& conda clean -a

