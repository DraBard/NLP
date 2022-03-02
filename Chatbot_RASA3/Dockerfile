FROM rasa/rasa:main-spacy-en

#Running as root to be able to modify the existing image
USER root

RUN pip3 install rasa[spacy]
RUN python -m spacy download en_core_web_sm

WORKDIR /app
COPY . .

RUN rasa train

# switch to safe user
USER 1001

#set entrypoint for interactive shells
ENTRYPOINT ["rasa"]

#command to run when container is called to run
CMD ["run", "--enable-api", "--port", "8080"]