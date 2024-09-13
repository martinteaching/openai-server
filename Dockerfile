FROM python:3.11
WORKDIR /openai-server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install .
CMD ['openaiserver']

