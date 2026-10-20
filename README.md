# spermoscope_backend

Flask-based AI inference and clinical database backend for the Spermoscope system.

## Features
- **WHO Semen Analysis & Strict Image Validation**: Powered by Google Gemini Vision AI (`gemini-3.1-flash-lite-preview` / `gemini-3-flash-preview` / `gemini-2.5-flash`). Detects and rejects unrelated or non-microscopic images with clinical explanations.
- **Database Architecture**: Powered by MongoDB Atlas with resilient fallback for full persistence of users, doctors, patient records, and multi-sample diagnostic results.
- **Doctor-Isolated Dashboards**: Each doctor accesses only their own registered patients; new doctors start with an initial clean/empty state.
- **RESTful Endpoints**: Authentication, patient intake, multi-sample tracking, and CASA diagnostic report APIs.

## Setup & Running

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```

3. **Start Server:**
   ```bash
   python app.py
   ```
