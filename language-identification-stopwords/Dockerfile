FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD runit.py /code/runit.py
ADD stopwords /code/stopwords
ADD model.joblib /code/model.joblib

ENTRYPOINT [ "python3", "/code/runit.py" ]
