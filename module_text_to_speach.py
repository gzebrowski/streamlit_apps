from io import BytesIO

import requests
import streamlit as st
from openai import OpenAI

from base_mod import BaseStreamlitApp
from streamlit_env import Env

MODELS = ["tts-1", "tts-1-hd"]
VOICES = ["alloy", "Ash", "Coral", "Echo", "Fable", "Onyx", "Nova", "Sage", "Shimmer"]
env = Env('.env')


def cut_text(text: str, max_chars: int = 4024):
    elipsis = ''
    if len(text) > max_chars:
        text = text[:max_chars]
        elipsis = '...'
    return f'{text}{elipsis}'


class TextToSpeachModule(BaseStreamlitApp):
    NAME = 'Text to speach'
    SESSION_DEFAULTS = [('api_key', None), ('last_files', []), ('idx', 0)]

    def write_html(self, html: str):
        st.write(html, unsafe_allow_html=True)

    @st.cache_resource
    def get_openai_client(_self):
        return OpenAI(api_key=_self.session_state.api_key)

    def _append_file(self, audio_bytes, filename, prompt=None):
        last_files = self.session_state.last_files or []
        curr_idx = (self.session_state.idx or 0) + 1
        self.session_state.idx = curr_idx
        last_files.insert(0, {'data': audio_bytes, 'filename': filename, 'prompt': prompt, 'idx': curr_idx})
        self.session_state.last_files = last_files

    def generate_voice(self, _openai_client, text: str, model: str = None, voice: str = None) -> str:
        response = _openai_client.audio.speech.create(
            model=model,
            voice=voice,
            response_format="mp3",
            input=text,
        )
        out_data = BytesIO(response.content)

        return out_data.getvalue()

    def get_main_content(self):
        tab1, = st.tabs(['Generuj mowę'])
        openai_client = self.get_openai_client()
        if self.get_api_key():
            with tab1:
                model = st.selectbox(label='Wybierz model', options=MODELS, key='model')
                prompt = st.text_area("wprowadź tekst", key='prompt1', max_chars=4024)
                voice = st.selectbox(label='Wybierz głos', options=VOICES, key='voice1')
                generate_btn = st.button('Potwierdź dane', key='generate_image_btn', disabled=not prompt)

                if prompt and voice and model and generate_btn:
                    with st.spinner('Proszę czekać, generowanie mowy'):
                        audio_bytes = self.generate_voice(openai_client, text=prompt, model=model, voice=voice)
                    filename = f'{model}_{voice}_{self.session_state.idx}.mp3'
                    self._append_file(audio_bytes, filename, prompt)
                    st.audio(audio_bytes, format='audio/mp3')
                    st.download_button(label='pobierz plik', data=audio_bytes,
                                       file_name=filename, key='download_btn_last')
            if self.session_state.last_files:
                for nr, f in enumerate(self.session_state.last_files):
                    if nr:
                        st.divider()
                    st.audio(f['data'], format='audio/mp3')
                    st.download_button(label='pobierz plik', data=f['data'],
                                       file_name=f['filename'], key=f"download_btn_{f['idx']}")

                    st.write(cut_text(f['prompt'], 256))
                    if st.button('usuń', key=f"remove_btn_{f['idx']}"):
                        last_files = self.session_state.last_files or []
                        last_files = [li for li in last_files if li['idx'] != f['idx']]
                        self.session_state.last_files = last_files
                        st.rerun()
