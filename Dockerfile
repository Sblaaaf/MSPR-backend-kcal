# Dockerfile pour l'API FastAPI JARMY
FROM python:3.12-slim

# Répertoire de travail
WORKDIR /app

# Copier requirements et installer les dépendances
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn "python-dotenv>=1.0"

# Copier tout le code
COPY . /app

# Exposer le port FastAPI
EXPOSE 8001

# Commande de démarrage (avec dev reload pour développement)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
