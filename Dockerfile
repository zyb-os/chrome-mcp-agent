FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV ORCHESTRATOR_URL=http://agent-orchestrator:8000
# Chrome extension connects to this port
EXPOSE 8765
CMD ["python", "main.py"]
