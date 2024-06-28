# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install required libraries
RUN apt-get update && apt-get install -y wget unzip \
  && pip install gdown \
  && gdown --id 1r_6-idFqe-9k8kEfwUEFDtmGsXv-e9xA -O local_model.zip && unzip local_model.zip -d /code/ && rm local_model.zip


RUN pip install transformers torch pandas

ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]
