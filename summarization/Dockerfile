FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run_bert.py /code/run_bert.py
RUN pip3 install transformers
RUN pip3 install torch

ENTRYPOINT [ "python3", "/code/run_bert.py" ]
