
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
