FROM gcr.io/preference-extraction/pref_extract_pysc2

WORKDIR /root
COPY ./ /root/

ENV SC2PATH=/StarCraftII
ENTRYPOINT ["python3", "train.py"]