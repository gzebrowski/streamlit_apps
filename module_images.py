import os
from urllib.parse import urlparse
from io import BytesIO

import requests
import streamlit as st
from openai import OpenAI

from base_mod import BaseStreamlitApp
from streamlit_env import Env

MODELS = ["dall-e-3", "dall-e-2"]
SIZES = ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
LLM_MODEL = 'gpt-4o-mini'
MAX_SIZE = 25 * 1024 * 1024
MAX_SIZE_EDIT_SRC = 4 * 1024 * 1024
# MAX_SIZE = 250
env = Env('.env')

cost_map = {
    "dall-e-2": {
        ('standard', '1024x1024'): 0.02,
        ('standard', '512x512'): 0.018,
        ('standard', '256x256'): 0.016,
    },
    "dall-e-3": {
        ('standard', '1024x1024'): 0.04,
        ('standard', '1792x1024'): 0.08,
        ('standard', '1024x1792'): 0.08,
        ('hd', '1024x1024'): 0.08,
        ('hd', '1792x1024'): 0.12,
        ('hd', '1024x1792'): 0.12,
    },
}

store_keys = {
    # 'segments': ['id', 'seek', 'start', 'end', 'text']
}


class GenerateImageModule(BaseStreamlitApp):
    NAME = 'Generate image'
    SESSION_DEFAULTS = [('api_key', None), ('last_images', []), ('idx', 0)]

    def write_html(self, html: str):
        st.write(html, unsafe_allow_html=True)

    def _get_image_restult(self, response):
        url = response.data[0].url
        url_data = urlparse(url)
        filename = os.path.basename(url_data.path)
        img_bytes = requests.get(url).content
        description = response.data[0].revised_prompt
        return img_bytes, filename, description

    def edit_image(self, _openai_client, prompt: str, model: str, input_image, mask_image, size: str = None):
        response = _openai_client.images.edit(
            model=model,
            image=input_image,
            mask=mask_image,
            prompt=prompt,
            n=1,
            size=size,
        )
        return self._get_image_restult(response)

    def variation_image(self, _openai_client, model: str, input_image, size: str = None):
        response = _openai_client.images.create_variation(
            model=model,
            image=input_image,
            n=1,
            size=size
        )
        return self._get_image_restult(response)

    def generate_image(self, _openai_client, prompt: str, size: str = None, quality: str = None,
                       model: str = None) -> str:
        response = _openai_client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        # print(response.usage)
        return self._get_image_restult(response)

    @st.cache_resource
    def get_openai_client(_self):
        return OpenAI(api_key=_self.session_state.api_key)

    def _append_image(self, image_bytes, filename, desc, prompt=None):
        last_images = self.session_state.last_images or []
        curr_idx = (self.session_state.idx or 0) + 1
        self.session_state.idx = curr_idx
        last_images.insert(0, {'data': image_bytes, 'filename': filename, 'prompt': prompt, 'description': desc,
                            'idx': curr_idx})
        self.session_state.last_images = last_images

    def get_main_content(self):
        sizes = SIZES
        qualities = ['standard', 'hd']
        tab1, tab2, tab3 = st.tabs(['Generowanie obrazów', 'Edycja obrazów', 'wariacje'])
        openai_client = self.get_openai_client()
        if self.get_api_key():
            with tab1:
                model = st.selectbox(label='Wybierz model', options=MODELS, key='model')
                prompt = st.text_input("wprowadź zapytanie", key='prompt1')
                if model == 'dall-e-3':
                    quality = st.selectbox(label='Wybierz jakość', options=qualities, key='quality1')
                else:
                    quality = 'standard'
                if model:
                    opt_sizes = [k[1] for k in cost_map[model].keys() if k[0] == 'standard']
                    size = st.selectbox(label='Wybierz rozmiar', options=opt_sizes, key='size1')
                else:
                    size = None
                if size and model and quality:
                    cost3 = cost_map[model][(quality, size)]
                    st.write(f'Koszt obrazka: ${cost3}')
                generate_btn = st.button('Potwierdź dane', key='generate_image_btn', disabled=not prompt)

                if prompt and size and quality and model and generate_btn:
                    with st.spinner('Proszę czekać, generowanie obrazka'):
                        image_bytes, filename, desc = self.generate_image(openai_client, prompt=prompt, size=size,
                                                                          quality=quality, model=model)
                    self._append_image(image_bytes, filename, desc, prompt)
                    st.image(image_bytes, caption='Ostatni wygenerowany')
                    st.download_button(label='pobierz obrazek', data=image_bytes,
                                       file_name=filename, key='download_btn_last}')
            with tab2:
                edit_img_src = st.file_uploader('Wczytaj obraz', type=['jpg', 'png', 'webp', 'gif', 'jpeg'],
                                                accept_multiple_files=False, key='edit_image', help='Max 4MB')
                edit_mask_src = st.file_uploader('Wczytaj maskę', type=['png'],
                                                 accept_multiple_files=False, key='edit_mask_image', help='Max 4MB')
                if edit_img_src and edit_mask_src:
                    bts_edit_img = BytesIO(edit_img_src.getvalue()).getvalue()
                    bts_mask_img = BytesIO(edit_mask_src.getvalue()).getvalue()

                    is_fine = len(bts_edit_img) <= MAX_SIZE_EDIT_SRC and len(bts_mask_img) <= MAX_SIZE_EDIT_SRC
                    if not is_fine:
                        st.error(f':red[Plik nie może być większy niż {MAX_SIZE_EDIT_SRC}]')
                    else:
                        prompt2 = st.text_input("wprowadź zapytanie", key='prompt2')
                        opt_sizes2 = [k[1] for k in cost_map["dall-e-2"].keys()]
                        size2 = st.selectbox(label='Wybierz rozmiar', options=opt_sizes2, key='size2')
                        if size2:
                            cost2 = cost_map["dall-e-2"][('standard', size2)]
                            st.write(f'Koszt obrazka: ${cost2}')
                        generate_btn_edit = st.button('Potwierdź dane', key='edit_image_btn', disabled=not prompt2)
                        if prompt2 and generate_btn_edit:
                            with st.spinner('Proszę czekać, generowanie obrazka'):
                                image_bytes2, filename2, desc2 = self.edit_image(
                                    openai_client, prompt=prompt2, size=size2, model="dall-e-2",
                                    input_image=bts_edit_img, mask_image=bts_mask_img)
                            self._append_image(image_bytes2, filename2, desc2, prompt2)

            with tab3:
                variation_img_src = st.file_uploader('Wczytaj obraz', type=['jpg', 'png', 'webp', 'gif', 'jpeg'],
                                                     accept_multiple_files=False, key='variation_img', help='Max 4MB')
                if variation_img_src:
                    bts_variation_img = BytesIO(variation_img_src.getvalue()).getvalue()

                    is_fine = len(bts_variation_img) <= MAX_SIZE_EDIT_SRC
                    if not is_fine:
                        st.error(f':red[Plik nie może być większy niż {MAX_SIZE_EDIT_SRC}]')
                    else:
                        opt_sizes3 = [k[1] for k in cost_map["dall-e-2"].keys()]
                        size3 = st.selectbox(label='Wybierz rozmiar', options=opt_sizes3, key='size3')
                        if size3:
                            cost3 = cost_map["dall-e-2"][('standard', size3)]
                            st.write(f'Koszt obrazka: ${cost3}')
                        generate_btn_variation = st.button('Potwierdź dane', key='var_image_btn')
                        if generate_btn_variation:
                            with st.spinner('Proszę czekać, generowanie obrazka'):
                                image_bytes3, filename3, desc3 = self.variation_image(
                                    openai_client, size=size3, model="dall-e-2",
                                    input_image=bts_variation_img)
                            self._append_image(image_bytes3, filename3, desc3)
            if self.session_state.last_images:
                for nr, img in enumerate(self.session_state.last_images):
                    if nr:
                        st.divider()
                    st.image(img['data'])
                    st.download_button(label='pobierz obrazek', data=img['data'],
                                       file_name=img['filename'], key=f"download_btn_{img['idx']}")
                    st.write(img['description'])
                    st.write(prompt)
                    if st.button('usuń', key=f"remove_btn_{img['idx']}"):
                        last_images = self.session_state.last_images or []
                        last_images = [li for li in last_images if li['idx'] != img['idx']]
                        self.session_state.last_images = last_images
                        st.rerun()
