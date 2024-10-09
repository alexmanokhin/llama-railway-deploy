FROM python:3.10-slim
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "main.py"]
