FROM python:3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get install -y --only-upgrade libsystemd0 libudev1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirments .

RUN pip install --no-cache-dir -r requirments

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "App 1.py", "--server.port=8501"]