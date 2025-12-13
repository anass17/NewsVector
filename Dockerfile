# Utiliser l'image officielle Airflow (stable)
FROM apache/airflow:2.10.3-python3.10

# Passer en root pour installer des dépendances système si nécessaire
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Revenir à l'utilisateur airflow
USER airflow

# Copier le requirements.txt
COPY requirements.txt /requirements.txt

# Installer les dépendances Python
# --no-cache-dir permet de réduire la taille de l'image
RUN pip install --no-cache-dir -r /requirements.txt