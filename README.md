# LangChain and Dify Integration Project

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `DIFY_API_KEY`: Your Dify API key
- `DIFY_API_URL`: Dify API endpoint
- `OPENAI_API_KEY`: Your OpenAI API key (if using LangChain with OpenAI)

### 3. Run the Project

```bash
python src/main.py
```

## Project Structure

```
.
├── src/              # Source code
├── config/           # Configuration files
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── README.md         # This file
```
