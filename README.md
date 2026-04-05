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

# Start services (PostgreSQL/pgAdmin + app container)
docker-compose up -d

# Check status
docker-compose ps
```

## Usage

### Run Scripts Inside Container
```bash
# Data exploration
# Default (full dataset, default sample for language detection 1000)
docker-compose exec app python src/preprocessing/explore_data.py

# Dataset with personalized sample
docker-compose exec app python src/preprocessing/explore_data.py --run-mode sample --sample-size 500

# Optional: customize language detection sampling independently
docker-compose exec app python src/preprocessing/explore_data.py --run-mode sample --sample-size 1000 --language-mode sample --language-sample-size 1000

# Skip language detection entirely
docker-compose exec app python src/preprocessing/explore_data.py --run-mode sample --sample-size 1000 --language-sample-size 0

# full exploration - might take around 3 hours
docker-compose exec app python src/preprocessing/explore_data.py --language-mode full

# Generate word clouds (raw data)
docker-compose exec app python src/visualization/wordclouds.py

# Data preprocessing (full dataset)
docker-compose exec app python src/preprocessing/preprocess_data.py --run-mode full

# Data preprocessing (quick sample)
docker-compose exec app python src/preprocessing/preprocess_data.py --run-mode sample --sample-size 500

# Translation (sample)
docker-compose exec app python src/preprocessing/translate_preprocessed.py --run-mode sample --sample-size 500

# Translation (full)
docker-compose exec app python src/preprocessing/translate_preprocessed.py --run-mode full

# Load preprocessed data into DB (full)
docker-compose exec app python src/preprocessing/load_preprocessed_to_db.py
docker-compose exec app python src/preprocessing/load_ecsf_to_db.py

# Load preprocessed data into DB (sample of job postings)
docker-compose exec app python src/preprocessing/load_preprocessed_to_db.py --run-mode sample --sample-size 500

# Create extraction tables (job_skill, ecsf_tks_text)
docker-compose exec app python src/extraction/create_extraction_tables.py

# Load extraction tables from DB data
docker-compose exec app python src/extraction/load_extraction_tables.py

# Create embedding/similarity tables
docker-compose exec app python src/embedding/create_embedding_tables.py

# Embed skills (SBERT)
docker-compose exec app python src/embedding/embed_skills.py

# Embed ECSF text (SBERT)
docker-compose exec app python src/embedding/embed_ecsf.py

# Compute skill to ECSF similarity
docker-compose exec app python src/similarity/compute_similarity.py

# Create mapping view (interpretable results)
docker-compose exec app python src/mapping/create_mapping_view.py

# Evaluation 
docker-compose exec app python src/similarity/evaluate_similarity.py

# Minimal webapp
docker-compose exec app python webapp/app.py
# Open: http://localhost:5000
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d euro_cyber_db
```

### pgAdmin Web UI

Open http://localhost:5050 and log in with:

- Email: value of `PGADMIN_DEFAULT_EMAIL` (default: `admin@local.dev`)
- Password: value of `PGADMIN_DEFAULT_PASSWORD` (default: `admin`)

Then add a new server in pgAdmin:

- Host name/address: `postgres`
- Port: `5432`
- Maintenance DB: `euro_cyber_db`
- Username: `postgres`
- Password: `postgres`

### Clean Restart
```bash
# Stop and remove all containers + volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d
```

This is a work in progress. It will be polished and with a friendlier documentation when finished!