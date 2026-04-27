# Euro Cyber Data - Master Thesis Pipeline

Dockerized Python pipeline that runs on PostgreSQL, processes job-posting and ECSF data, extracts and maps skills with S-BERT embeddings, evaluates results with similarity metrics, and exposes a minimal Flask webapp for inspection.

## 🧱 Project Structure

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

## 🛠️ Tech Stack

- **Python 3.11** - Core language
- **PostgreSQL 15** - Database
- **Docker & Docker Compose** - Containerization
- **S-BERT** (sentence-transformers) - Multilingual embeddings
- **Flask** - Minimal web dashboard

## ⚙️ Setup

### 1. 🌱 Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env if needed (DB credentials, model names, etc.)
```

### 2. 🚀 Build and Start Containers
```bash
# Build Docker images
docker-compose build

# Start services (PostgreSQL/pgAdmin + app container)
docker-compose up -d

# Check status
docker-compose ps
```

## ▶️ Usage

### 🧪 Run Scripts Inside Container

Most preprocessing scripts support the same execution pattern:

- `--run-mode sample` runs on a smaller subset for quick iteration.
- `--sample-size` controls how many records are used in sample mode.
- `--run-mode full` runs on the complete dataset.
- Scripts that do not expose these flags always run on their full/default input.

Use sample mode first when you are checking logic, timings, or output shape. Switch to full mode only when you are ready for the complete pipeline run.

#### 🔎 Sample-First Commands
```bash
# Data exploration on a small subset
docker-compose exec app python src/preprocessing/explore_data.py --run-mode sample --sample-size 500

# Optional: customize language detection sampling independently
docker-compose exec app python src/preprocessing/explore_data.py --run-mode sample --sample-size 1000 --language-mode sample --language-sample-size 1000

# Data preprocessing
docker-compose exec app python src/preprocessing/preprocess_data.py --run-mode sample --sample-size 500

# Translation
docker-compose exec app python src/preprocessing/translate_preprocessed.py --run-mode sample --sample-size 500

# Load preprocessed job postings into DB
docker-compose exec app python src/preprocessing/load_preprocessed_to_db.py --run-mode sample --sample-size 500
```

#### 🧾 Full-Run Commands
```bash
# Full data exploration
docker-compose exec app python src/preprocessing/explore_data.py --language-mode full

# Generate word clouds (over raw data)
docker-compose exec app python src/visualization/wordclouds.py

# Data preprocessing
docker-compose exec app python src/preprocessing/preprocess_data.py --run-mode full

# Translation
docker-compose exec app python src/preprocessing/translate_preprocessed.py --run-mode full

# Load preprocessed data into DB
docker-compose exec app python src/preprocessing/load_preprocessed_to_db.py
docker-compose exec app python src/preprocessing/load_ecsf_to_db.py

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

### 🗄️ Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d euro_cyber_db
```

### 🌐 pgAdmin Web UI

Open http://localhost:5050 and log in with:

- Email: value of `PGADMIN_DEFAULT_EMAIL` (default: `admin@local.dev`)
- Password: value of `PGADMIN_DEFAULT_PASSWORD` (default: `admin`)

Then add a new server in pgAdmin:

- Host name/address: `postgres`
- Port: `5432`
- Maintenance DB: `euro_cyber_db`
- Username: `postgres`
- Password: `postgres`

### 🔄 Clean Restart
```bash
# Stop and remove all containers + volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d
```