# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN apt-get update && apt-get install -y wget unzip
RUN wget -O local_model.zip 'https://drive.google.com/file/d/1r_6-idFqe-9k8kEfwUEFDtmGsXv-e9xA/view?usp=sharing' && unzip local_model.zip -d /code/local_model && rm local_model.zip


RUN pip install transformers torch pandas os

ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]
