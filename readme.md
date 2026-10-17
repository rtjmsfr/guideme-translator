# 🌍 FastAPI Multilingual Translation API (M2M100)

This project is a FastAPI-based RESTful API that uses Facebook AI’s [M2M100 multilingual model](https://huggingface.co/facebook/m2m100_418M) from Hugging Face Transformers to perform translation across **100+ languages** — all without using English as an intermediary.

---

## 🚀 Features

* Translate text between any two supported languages
* Batch translation (multiple input sentences at once)
* Language code lookup endpoint (`/languages`)
* CORS-enabled for integration with frontend apps
* Hugging Face token support for private model access and caching

---

## ⚠️ Note on First-Time Use

> **⏳ The first time you run the API, it may take a few minutes to start.**
> This is because the M2M100 model and tokenizer (approx. 1.2 GB) will be downloaded and cached locally in the `./m2m_model` folder.

Subsequent runs will use the cached model, so startup will be much faster.

---

## 📦 Requirements

* Python 3.12.3+
* Conda (recommended)
* Hugging Face account + access token

---

## 🔧 Installation

### 1. Create Conda environment

```bash
conda create -n fastapi-translator python=3.12.3 -y
conda activate fastapi-translator
```

### 3. Install Python dependencies

```bash
pip install fastapi uvicorn transformers torch pydantic[dotenv]
```

---

## 🔑 Add your Hugging Face token

Set the `HF_TOKEN` environment variable before running:

```bash
export HF_TOKEN="hf_your_token_here"
```

Get your own **Hugging Face API token** (get one here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

---

## 💻 Run the Server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Or simply run:

```bash
python main.py
```

Server will be running at:

```bash
http://127.0.0.1:8000
```

---

## 📮 API Endpoints

### `POST /translate`

Translate one or more texts from a source language to a target language.

#### Request

```json
{
  "text": ["Hello, how are you?", "Nice to meet you."],
  "source_lang": "en",
  "target_lang": "fr"
}
```

#### Response

```json
{
  "translations": [
    {
      "text": "Hello, how are you?",
      "translated": "Bonjour, comment ça va ?"
    },
    {
      "text": "Nice to meet you.",
      "translated": "Ravi de vous rencontrer."
    }
  ]
}
```

---

### `GET /languages`

Returns a dictionary of supported language codes and their full names.

#### Sample Response

```json
{
  "en": "English",
  "fr": "French",
  "es": "Spanish",
  ...
}
```

---

## 📚 Language Codes

The API uses ISO 639-1 language codes. You can view the full list by calling `/languages`.

Examples:

| Language             | Code |
| -------------------- | ---- |
| English              | `en` |
| French               | `fr` |
| Arabic               | `ar` |
| Chinese (Simplified) | `zh` |
| Russian              | `ru` |

---
## 🗂 Directory Structure

```css
guideme-translator/
├── main.py
├── requirements.txt
├── README.md
└── m2m_model/         # Model cache (created on first run, git-ignored)
```
