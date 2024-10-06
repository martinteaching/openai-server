FROM python:latest
WORKDIR /openai-server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install .
CMD ['openaiserver']

