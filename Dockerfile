# Gunakan image Python
FROM python:3.11.5

# Set workdir
WORKDIR /app

# Copy file requirement.txt ke container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke container
COPY . .

# Expose port yang akan digunakan FastAPI
EXPOSE 8080

# Jalankan perintah untuk menjalankan server FastAPI
CMD ["uvicorn", "main:app", "--reload", "--port=8080", "--host=0.0.0.0"]