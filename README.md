# CVMatch AI 🎯

> An intelligent CV-to-job-offer matching assistant powered by Large Language Models.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-WIP-yellow.svg)]()

## 🎯 What it does

CVMatch AI helps job seekers optimize their applications by:

- **Analyzing** how well their CV matches a given job offer
- **Identifying** missing keywords and skill gaps
- **Rewriting** the CV to better align with the target role
- **Generating** a personalized cover letter
- **Predicting** likely interview questions based on the role

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **LLM Provider**: Groq (free tier, openai/gpt-oss-20b)
- **API Pattern**: OpenAI-compatible SDK (provider-agnostic)
- **Frontend**: Streamlit (coming soon)
- **PDF Parsing**: pdfplumber (coming soon)

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/Lidelle123/cvmatch-ai.git
cd cvmatch-ai

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (get one at https://console.groq.com/keys)

# Run the demo
python hello_llm.py
```

## 📅 Roadmap

- [x] **Day 1**: LLM API setup + multi-provider abstraction
- [ ] **Day 2**: Structured outputs with Pydantic (CV parsing)
- [ ] **Day 3**: PDF extraction pipeline
- [ ] **Day 4**: Job offer parsing & matching score
- [ ] **Day 5+**: Streamlit UI, deployment, advanced features

## 👤 Author

**Vanella Lidelle Dzikang** — AI Engineer in training  
[LinkedIn](https://www.linkedin.com/in/vanella-dzikang-9a4924288/) · [GitHub](https://github.com/Lidelle123)

## 📄 License

MIT
