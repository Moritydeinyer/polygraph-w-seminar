FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=80", "--server.address=0.0.0.0"]