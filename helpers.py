from hashlib import md5
from io import BytesIO
from typing import Any, Union

import streamlit as st

AUDIO_TRANSCRIBE_MODEL = "whisper-1"


def write_html(html: str):
    st.write(html, unsafe_allow_html=True)


def write_small(text: str):
    write_html(f'<p style="color:#888; font-size:0.7em;"><em>{text}</em></p>')


def to_int(val: Union[str, int, float], default: Any = None) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def transcribe_audio(openai_client, audio_bytes: bytes, language='pl') -> str:
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
        language=language,
    )
    return transcript.text


def get_mp3_audio_and_hash(rec_data):
    audio = BytesIO()
    rec_data.export(audio, format="mp3")
    audio_bytes = audio.getvalue()
    hsh = md5(audio_bytes).hexdigest()
    return audio_bytes, hsh


def is_numeric(val):
    try:
        if val is not None:
            float(val)
    except (TypeError, ValueError):
        return False
    return True
