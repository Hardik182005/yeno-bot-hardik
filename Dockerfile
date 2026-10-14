FROM python:3.12-slim

WORKDIR /app
COPY bot/yeno_bot.py /app/yeno_bot.py

ENV HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
USER nobody
CMD ["python3", "/app/yeno_bot.py"]
