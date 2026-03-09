# Euro Cyber Data - Master Thesis Pipeline

Cybersecurity skills mapping: Job Market → ECSF using S-BERT embeddings

## Project Structure

```
euro-cyber-data/
├── data/
│   ├── raw/              # ecsf.json, job_postings.json
│   └── preprocessed/     # Preprocessed data
├── src/
│   ├── preprocessing/    # Data exploration & preprocessing
│   ├── extraction/       # Skill extraction
│   ├── embedding/        # Embedding generation (S-BERT)
│   ├── similarity/       # Similarity computation
│   ├── mapping/          # Final mapping logic
│   └── utils/            # Utilities (DB, config)
├── database/
│   └── schema.sql        # PostgreSQL schema
├── webapp/               # Minimal Flask dashboard (placeholder)
├── docker/
│   └── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack

- **Python 3.11** - Core language
- **PostgreSQL 15** - Database
- **Docker & Docker Compose** - Containerization
- **S-BERT** (sentence-transformers) - Multilingual embeddings
- **Flask** - Minimal web dashboard

## Setup

### 1. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env if needed (DB credentials, model names, etc.)
```

### 2. Build and Start Containers
```bash
# Build Docker images
docker-compose build

# Start services (PostgreSQL + app container)
docker-compose up -d

# Check status
docker-compose ps
```

## Usage

### Run Scripts Inside Container
```bash
# Data exploration
docker-compose exec app python src/preprocessing/explore_data.py

# Data preprocessing
docker-compose exec app python src/preprocessing/preprocess_data.py
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d euro_cyber_db
```

### Clean Restart
```bash
# Stop and remove all containers + volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d
```

This is a work in progress. It will be polished and with a friendlier documentation when finished!