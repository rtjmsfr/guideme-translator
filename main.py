import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import List
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


token = os.getenv("HF_TOKEN")

model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M", token=token, cache_dir="./m2m_model"
)
tokenizer = M2M100Tokenizer.from_pretrained(
    "facebook/m2m100_418M", token=token, cache_dir="./m2m_model"
)

languages = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "ast": "Asturian", "az": "Azerbaijani", "be": "Belarusian",
    "bg": "Bulgarian", "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "ceb": "Cebuano",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian (Farsi)", "fi": "Finnish", "fr": "French", "fy": "Western Frisian", "ga": "Irish",
    "gd": "Scottish Gaelic", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "ig": "Igbo",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh",
    "km": "Central Khmer", "kn": "Kannada", "ko": "Korean", "lb": "Luxembourgish", "lg": "Ganda", "ln": "Lingala",
    "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "ns": "Northern Sotho", "oc": "Occitan", "or": "Oriya", "pa": "Panjabi",
    "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "qu": "Quechua", "ro": "Romanian", "ru": "Russian",
    "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "ti": "Tigrinya", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "wo": "Wolof", "xh": "Xhosa", "yi": "Yiddish",
    "yo": "Yoruba", "zh": "Chinese (Simplified)", "zh-Hant": "Chinese (Traditional)", "zu": "Zulu"
}




class TranslationRequest(BaseModel):
    text: List[str]
    source_lang: str
    target_lang: str


@app.post("/translate")
def translate(req: TranslationRequest):
    tokenizer.src_lang = req.source_lang
    results = []

    for t in req.text:
        encoded = tokenizer(t, return_tensors="pt")
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(req.target_lang)
        )
        translated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        results.append({
            "text": t,
            "translated": translated_text
        })

    return {"translations": results}


@app.get("/languages")
def get_languages():
    return languages

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5050, reload=False)
