# Avatar Content Ranking System

This system uses OpenRouter's multimodal AI models to analyze and rank avatar images for roleplay and virtual dating content.

## Features

- **Structured Output**: Uses OpenRouter's structured output feature with strict JSON schema validation
- **Multimodal Analysis**: Processes images using Claude 3.5 Sonnet (latest) with vision capabilities
- **Database Integration**: Queries PostgreSQL database and optionally updates rankings
- **Comprehensive Scoring**: 0-2000 point scoring system based on appeal, quality, and artistic merit
- **Classification**: Automatically categorizes gender (male/female/non-binary) and style (stylized/realistic)
- **Flexible Execution**: Run with or without database writes via command line flags

## Setup

### 1. Credentials
Add your credentials to `Rita_LLM/.env.local`:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
POSTGRES_URL=your_postgres_connection_string_here
```

### 2. Run the Ranking System

**Local analysis only (no database writes):**
```bash
python Rita_LLM/content_rank/rank.py
```

**With database updates:**
```bash
python Rita_LLM/content_rank/rank.py --write-db
```

**Process more avatars:**
```bash
python Rita_LLM/content_rank/rank.py --max-avatars 10 --write-db
```

## Command Line Options

- `--max-avatars N`: Process up to N avatars (default: 3)
- `--write-db`: Write results back to database (default: local only)

## How It Works

1. **Database Query**: Retrieves recent public avatars with image URLs
2. **AI Analysis**: Downloads and analyzes images using Claude 3.5 Sonnet
3. **Local Storage**: Saves results to `Rita_LLM/content_rank/output/{avatar_id}/result.txt`
4. **Optional DB Update**: Updates avatars table with scores/classifications (if `--write-db` flag is used)

## Output Format

Each avatar gets:
- **Score**: Integer from 0-2000 based on comprehensive criteria
- **Gender**: male, female, or non-binary classification  
- **Style**: stylized or realistic classification

Results are always saved locally, and optionally written to database.

## Scoring Criteria

**Additions (increases score):**
- Stylized content: +100 flat bonus
- Revealing/sexy clothing: +100 × coefficient (1.00-10.00)
- Youth appeal (18-22): +50 × coefficient (1.00-10.00) 
- Artistic design: +50 × coefficient (1.00-10.00)

**Subtractions (decreases score):**
- Lacks appealing features: -100
- Poor quality/AI artifacts: -200 × coefficient (1.00-10.00)
- Dull/unengaging: -100 × coefficient (1.00-10.00)

## Technical Details

- **Model**: `anthropic/claude-3.5-sonnet-20241022` (latest)
- **Credentials**: Loaded from `Rita_LLM/.env.local`
- **Output Directory**: `Rita_LLM/content_rank/output/`
- **Temperature**: 0.1 for consistent results
- **Timeouts**: 60s for API, 30s for downloads 