FROM gcr.io/preference-extraction/pref_extract_tf_torch

RUN echo 'Launching training on GCP V1.9.1'

WORKDIR /root
COPY ./ /root/

ENTRYPOINT ["python3", "extract_preferences.py"]