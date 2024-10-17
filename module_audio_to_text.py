import json
import re
from hashlib import md5, sha1
from io import BytesIO
from typing import Optional

import streamlit as st
from openai import OpenAI
from st_click_detector import click_detector
from streamlit_player import st_player

from base_mod import BaseStreamlitApp
from streamlit_env import Env

AUDIO_TRANSCRIBE_MODEL = "whisper-1"
LLM_MODEL = 'gpt-4o-mini'
MAX_SIZE = 25 * 1024 * 1024
# MAX_SIZE = 250
env = Env('.env')


store_keys = {
    # 'segments': ['id', 'seek', 'start', 'end', 'text']
}


class AudioToTextModule(BaseStreamlitApp):
    NAME = 'Audio to text'
    SESSION_DEFAULTS = [('api_key', None), ('file_key', 1), ('file_hash', None), ('text', ''), ('words', ''),
                        ('segments', ''), ('uploaded_filename', ''), ('video_offset', 0), ('clicked', 1),
                        ('summary', ''), ('sum_text', ''), ('sum_summary', ''), ('sum_key', 1), ('sum_load_text', ''),
                        ('sum_file', ''), ('ff_sum_text', ''), ('ff_sum_key', 1)]

    def write_html(self, html: str):
        st.write(html, unsafe_allow_html=True)

    def format_tm(self, tm):
        tm = int(tm)
        parts = [tm // 3600, (tm % 3600) // 60, tm % 60]
        for _ in range(2):
            parts = parts[1:] if not parts[0] else parts
        return ':'.join([f"{nr:02d}" for nr in parts])

    def transcribe_audio(self, _openai_client, audio_bytes: bytes, word_timestamp=False,
                         language: Optional[str] = None) -> str:
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        timestamp_granularities = ["segment"]
        if word_timestamp:
            timestamp_granularities = ['word'] + timestamp_granularities
        transcript_kws = {'language': language} if language else {}
        transcript = _openai_client.audio.transcriptions.create(
            file=audio_file,
            model=AUDIO_TRANSCRIBE_MODEL,
            response_format="verbose_json",
            timestamp_granularities=timestamp_granularities,
            **transcript_kws,
        )

        words = getattr(transcript, 'words', [])
        segments = getattr(transcript, 'segments', [])
        return transcript.text, words, segments, [words, segments]

    def get_audio_hash(self, audio_bytes):
        _hsh = md5(audio_bytes).hexdigest()
        return _hsh

    def set_video_offset(self, offset=0):
        self.session_state.video_offset = offset

    def get_segment_lines(self, segments, pattern):
        _html_lines = []
        for _nr, _line in enumerate(segments):
            line2 = dict(_line)
            for k3 in ['start', 'end']:
                line2[f'{k3}2'] = self.format_tm(line2[k3])
            s_line = {'time': int(line2['start']), 'id': _nr}
            line2['int_tm'] = s_line['time']
            s_line['html'] = pattern % ({'id': _nr} | line2)
            _html_lines.append(s_line)
        return _html_lines

    def get_summary(self, _openai_client, long_text: str) -> str:
        with st.spinner('Proszę czekać, tworzenie podsumowania'):
            prompt2 = 'Wygeneruj streszczenie poniższej treści. '\
                'Jeśli to możliwe, to streszczenie powinno być w formie wypunktowania głównych zagadnien '\
                'zawartych w treści :\n\n'
            prompt2 += long_text
            response = _openai_client.chat.completions.create(model=LLM_MODEL, temperature=0, messages=[{
                'role': 'user',
                'content': [{'type': 'text', 'text': prompt2}],
            }])
            return response.choices[0].message.content

    @st.cache_resource
    def get_openai_client(self):
        return OpenAI(api_key=self.session_state.api_key)

    def get_main_content(self):
        can_process = True
        allowed_emails = env.get('ALLOWED_EMAILS')
        if allowed_emails:
            can_process = False
            email = st.text_input('Podaj swój email')
            allowed_emails = allowed_emails.split(',')
            if email in allowed_emails:
                can_process = True

        if not self.session_state.api_key:
            api_key = st.text_input('Podaj klucz api', type='password')
            if api_key:
                hsh = env.get('PWD_HASH')
                salt = env.get('PWD_SALT')
                env_api_key = env.get('OPENAI_API_KEY')
                if hsh and salt and env_api_key:
                    if sha1(f'{salt} {api_key}'.encode()).hexdigest() == hsh:
                        api_key = env_api_key
                self.session_state.api_key = api_key
                st.rerun()
        else:
            f_key = f"k_{self.session_state['file_key']}"
            tab1, tab2, tab3, tab4, tab5 = st.tabs(['Przetwarzanie audio', 'prezentacja', 'generuj streszczenie',
                                                    'odczyt streszczenia', 'logs'])
            html_lines = []

            with tab1:
                uploaded_file = st.file_uploader('Wczytaj plik mp3', type=['mp3'], accept_multiple_files=False,
                                                 key=f_key, help='Max 25MB')
                lang_map = {'': 'Auto', 'pl': 'Polski', 'en': 'Angielski'}
                sel_language = st.selectbox('Wybierz język', options=['', 'pl', 'en'],
                                            format_func=lambda x: lang_map.get(x, x))
                fetch_word_timestamps = st.checkbox('Pobierz timestamp wszystkich wyrazów')
                sumarize_text = st.checkbox('Utwórz streszczenie treści')
                if can_process and uploaded_file:
                    bts = BytesIO(uploaded_file.getvalue()).getvalue()
                    is_fine = len(bts) <= MAX_SIZE
                    if not is_fine:
                        st.error(f':red[Plik nie może być większy niż {MAX_SIZE}]')
                        ok_error = st.button('OK', type='primary')
                        if ok_error:
                            self.session_state.file_key = self.session_state['file_key'] + 1
                            st.rerun()
                    else:
                        summary = self.session_state.summary
                        uploaded_filename = uploaded_file.name
                        file_hash = self.get_audio_hash(bts)
                        openai_client = self.get_openai_client()
                        if file_hash != self.session_state.file_hash:
                            self.session_state.file_hash = file_hash
                            self.session_state.uploaded_filename = uploaded_filename
                            with st.spinner('Proszę czekać, przetwarzanie dźwięku'):
                                data = self.transcribe_audio(openai_client, bts, word_timestamp=fetch_word_timestamps,
                                                             language=sel_language)
                            if env.get('LOG_DATA'):
                                self.log(data[3])
                            summary = ''
                            if sumarize_text:
                                summary = self.get_summary(openai_client, data[0])
                                self.session_state.summary = summary
                            for nr, k in enumerate(['text', 'words', 'segments']):
                                dt = data[nr]
                                if dt and k in store_keys and isinstance(dt, list) and isinstance(dt[0], dict):
                                    dt = [{k2: v2 for k2, v2 in itm.items() if k2 in store_keys[k]} for itm in dt]
                                self.session_state[k] = dt
                        for k in ['text', 'words', 'segments']:
                            st.header(k)
                            with st.container(height=200):
                                st.write(self.session_state[k])
                            if self.session_state[k]:
                                ext = 'txt' if k == 'text' else 'json'
                                dt2store = json.dumps(self.session_state[k]) if k in [
                                    'words', 'segments'] else str(self.session_state[k])
                                st.download_button(label=f'Pobierz {k}', data=dt2store,
                                                   file_name=f'{self.session_state.uploaded_filename}_{k}.{ext}')
                        if not summary and self.session_state.text:
                            gen_summary_btn = st.button('Generuj streszczenie', key=f'gen_summary_btn_{file_hash}')
                            if gen_summary_btn:
                                summary = self.get_summary(openai_client, self.session_state.text)
                                self.session_state.summary = summary

                        if summary:
                            st.header('Streszczenie treści')
                            with st.container(height=200):
                                st.write(summary)
                            st.download_button(label='Pobierz streszczenie', data=summary,
                                               file_name=f'{self.session_state.uploaded_filename}_summary.md')

                        download_html = st.button('Spreparuj html', key=f'html_{f_key}', help='Pobierz tekst jako html')
                        if download_html:
                            p = '<p id="p_%(id)s" title="start: %(start2)s, end: %(end2)s" class="txt_line">'\
                                '<span class="timestamp" data-tm="%(int_tm)s">(%(start2)s - %(end2)s)</span>'\
                                '\n%(text)s\n'\
                                '</p>'
                            html_lines = self.get_segment_lines(self.session_state['segments'], p)
                            jquery = open('jquery.js').read()
                            tpl = open('html_template.txt', encoding='utf8').read()
                            full_html = tpl % {'title': self.session_state.uploaded_filename, 'jquery': jquery,
                                               'body': '\n'.join([h['html'] for h in html_lines])}
                            st.download_button(label='Pobierz html', data=full_html,
                                               file_name=f'{self.session_state.uploaded_filename}.html')

                        reset_all = st.button('Zresetuj', key=f'btn_{f_key}', type='primary',
                                              help='Umożliwia wczytanie innego pliku')
                        if reset_all:
                            self.session_state.file_key = self.session_state['file_key'] + 1
                            for k in ['text', 'words', 'segments']:
                                self.session_state[k] = ''
                                st.rerun()
            with tab2:
                if not self.session_state.segments:
                    load_segments = st.file_uploader('Załaduj json z danymi "segments"', key=f'segm_{f_key}',
                                                     type=['json'])
                    if load_segments:
                        self.session_state.segments = json.loads(load_segments.getvalue())
                        st.rerun()
                else:
                    url = st.text_input('URL z video na YT', on_change=self.set_video_offset)
                    yt_src = ''
                    if url:
                        if url and re.match(r'^https?://(www\.youtube\.com|youtu\.be)/[a-zA-Z0-9_.+=,#\/?&%-]+$', url):
                            key_yt = re.match(r'^https?://[^/]+/embed/([a-zA-Z0-9_.-]+)', url)
                            key_yt = key_yt or re.match(r'^https?://[^?]+.*?[?&]v=([a-zA-Z0-9_.-]+)', url)
                            key_yt = key_yt[1] if key_yt else None
                            start_yt = self.session_state.video_offset or 0
                            yt_src = f'http://www.youtube.com/embed/{key_yt}&start={start_yt}' if key_yt else ''
                    col1, col2 = st.columns(2)
                    clicked = {}
                    with col1:
                        htm_p = '<p title="start: %(start2)s, end: %(end2)s">%(text)s</p>'
                        html_lines = self.get_segment_lines(self.session_state['segments'], htm_p)
                        with st.container(height=600):
                            for line in html_lines:
                                self.write_html(line['html'])
                                if yt_src:
                                    c_key = self.session_state['clicked']
                                    clicked[line['id']] = click_detector(
                                        f'''<a href="#" id="clck_{line['id']}">-&gt;</a>''',
                                        key=f"goto_{c_key}_{line['id']}")
                                    if clicked[line['id']]:
                                        self.session_state.clicked = self.session_state['clicked'] + 1
                                        self.set_video_offset(offset=line['time'])
                                        st.rerun()
                    with col2:
                        if yt_src:
                            kws = {'playing': True} if self.session_state.video_offset else {}
                            st_player(yt_src, **kws)
            with tab3:
                sum_text_kws = {'value': self.session_state.sum_load_text} if self.session_state.sum_load_text else {}
                sum_text = st.text_area('Wklej tekst do podsumowania', key='sum_text2', **sum_text_kws)
                load_txt = st.file_uploader('Lub załaduj z pliku', key=f'txt_from_file_{self.session_state.sum_key}',
                                            type=['txt'])
                if load_txt:
                    self.session_state.sum_key += 1
                    self.session_state.sum_load_text = load_txt.getvalue().decode()
                    st.rerun()

                if sum_text:
                    if sum_text != self.session_state.sum_text:
                        sum_btn = st.button('Generuj podsumowanie', key=f'sum_btn2_{self.session_state.sum_key}')
                        if sum_btn:
                            self.session_state.sum_key += 1
                            self.session_state.sum_text = sum_text
                            openai_client = self.get_openai_client()
                            self.session_state.sum_summary = self.get_summary(openai_client, sum_text)
                if self.session_state.sum_summary:
                    st.header('Streszczenie treści')
                    with st.container(height=200):
                        st.write(self.session_state.sum_summary)
                    st.download_button(label='Pobierz streszczenie', data=self.session_state.sum_summary,
                                       file_name='summary.md')

            with tab4:
                load_summary = st.file_uploader('Załaduj z pliku', key=f'sum_from_file_{self.session_state.ff_sum_key}',
                                                type=['md'])
                if load_summary:
                    self.session_state.ff_sum_key += 1
                    self.session_state.ff_sum_text = load_summary.getvalue().decode()
                if self.session_state.ff_sum_text:
                    st.write(self.session_state.ff_sum_text)

            with tab5:
                for _log in self.get_logs():
                    st.write(_log)
