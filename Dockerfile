FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK & NumPy (if not in requirements.txt already)
RUN python -m pip install --no-cache-dir nltk numpy

# Download NLTK data into /usr/local/nltk_data
RUN python -m nltk.downloader punkt vader_lexicon stopwords -d /usr/local/nltk_data

# Copy app code
COPY . .

# Tell NLTK to look in that folder at runtime
ENV NLTK_DATA=/usr/local/nltk_data

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
