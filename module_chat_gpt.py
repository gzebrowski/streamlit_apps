import base64
import datetime
import mimetypes
import sqlite3
from io import BytesIO
from typing import Optional

import streamlit as st
from pydantic import Field

from base_mod import BaseStreamlitApp, env, get_openai_client
from tiny_orm import AbstractModel, prepare_db, register_model


@st.cache_resource
def connect_db():
    connection = sqlite3.connect('custom_chat_gpt.db')
    connection.row_factory = sqlite3.Row
    return connection


def smart_date(dt: datetime.date) -> str:
    """
    :param dt: datetime.date or datetime.datetime
    :return: string representing smart date
    Smart date formatter, that returns string in format depending on how old is that passed date and if the passed
    date is time aware.
    """
    now = datetime.datetime.now()
    today = now.date()
    formatters = {
        ('today', True): '%H:%M',
        ('today', False): 'dzisiaj',
        ('yesterday', True): 'wczoraj, %H:%M',
        ('yesterday', False): 'wczoraj',
        ('date', True): '%d %b, %H:%M',
        ('date', False): '%d %b',
        ('old_date', True): '%d %b',
        ('old_date', False): '%d %b',
        ('very_old_date', True): '%d %b %Y',
        ('very_old_date', False): '%d %b %Y',
    }
    is_datetime = True
    if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        the_date = dt
        is_datetime = False
    else:
        the_date = dt.date()
    key = 'very_old_date'
    if the_date == today:
        key = 'today'
    elif the_date == today - datetime.timedelta(days=1):
        key = 'yesterday'
    elif the_date > today - datetime.timedelta(days=30):
        key = 'date'
    elif the_date > today - datetime.timedelta(days=365):
        key = 'old_date'
    formatter = formatters.get((key, is_datetime))
    return dt.strftime(formatter)


@register_model
class Threads(AbstractModel):
    model: str
    personality_id: Optional[int] = Field(json_schema_extra={'Relation': 'Personality'})
    name: str
    created_at: datetime.datetime = Field(json_schema_extra={'auto_now_add': True})

    @classmethod
    def _get_connection(cls):
        return connect_db()


@register_model
class Personality(AbstractModel):
    name: str
    value: str

    @classmethod
    def _get_connection(cls):
        return connect_db()


@register_model
class Messages(AbstractModel):
    thread_id: int = Field(json_schema_extra={'Relation': 'Threads'})
    created_at: datetime.datetime = Field(json_schema_extra={'auto_now_add': True})
    message: str
    role: str
    cost: Optional[float]
    total_cost: Optional[float]
    completion_tokens: Optional[float]
    prompt_tokens: Optional[float]
    total_tokens: Optional[float]

    @classmethod
    def _get_connection(cls):
        return connect_db()


class ChatGPTModule(BaseStreamlitApp):
    SHOW_SIDEBAR = True
    SESSION_DEFAULTS = [('messages', []), ('cache_total', 0), ('cache_cost', 0)]
    NAME = 'Chat GPT'

    def __init__(self):
        super().__init__()
        self.sel_thread = None
        self.history_size = None
        prepare_db()
        self.chat_opts = ['gpt-4o', 'gpt-4o-mini']
        self.model_pricings = {}
        for model_opt in self.chat_opts:
            model_opt_key = model_opt.upper().replace('-', '_')
            self.model_pricings[model_opt] = {
                'input_tokens': env[f'PRICING_{model_opt_key}_INPUT'],
                'output_tokens': env[f'PRICING_{model_opt_key}_OUTPUT'],
            }
        db_threads = list(Threads.objects().order_by('-id'))
        self.thrs_data = {db_thread.name: db_thread for db_thread in db_threads}
        all_pers = list(Personality.objects().order_by('id'))
        self.all_pers_idxs = {p.id: nr3 for nr3, p in enumerate(all_pers)}
        self.pers_names = {p.name: (p.pk, p.value) for p in all_pers}

    def get_messages(self, thread_id, limit):
        msgs = list(Messages.objects().filter(thread_id=thread_id).order_by('-id')[:limit])[::-1]
        result = [{'role': m.role, 'message': m.message,
                   'timestamp': m.created_at, 'cost': m.cost} for m in msgs]
        self.session_state.messages = result
        self.session_state.cache_total = max([m.total_cost for m in msgs if m.total_cost] + [0])
        self.session_state.cache_cost = (msgs[-1].cost if msgs else 0) or 0
        return result

    def remove_personality(self, pers_id):
        connection = connect_db()
        Personality.objects(connection=connection).filter(id=pers_id).delete()
        self.session_state.num = self.session_state.num + 1

    def get_file_in_b64(self, file_data):
        mmtp = mimetypes.guess_type(file_data[0])
        if not mmtp or not mmtp[0]:
            return None, None
        encoded_file = base64.b64encode(file_data[1].read()).decode('utf-8')
        media_type = mmtp[0].split('/', 1)[0]
        return f"data:{mmtp[0]};base64,{encoded_file}", media_type

    def add_message(self, prompt, user_type, file=None):
        if self.sel_thread:
            Messages.create(thread_id=self.sel_thread.id, message=prompt, role=user_type)
            all_messages = list(Messages.objects().filter(thread_id=self.sel_thread.id).order_by('-id')[:self.history_size])
            msgs2send = [{'role': m.role, 'content': m.message} for m in all_messages][::-1]
            msgs2send = [
                {
                    "role": "system",
                    "content": self.sel_thread.personality.value,
                },
            ] + msgs2send

            if file:
                b64_file, media_type = self.get_file_in_b64(file)
                if b64_file and media_type in ['image', 'audio']:
                    file_type = f'{media_type}_url'
                    update_msg = msgs2send.pop()
                    update_msg['content'] = [
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                        {
                            'type': file_type,
                            file_type: {
                                'url': b64_file,
                                'detail': 'high',
                            },
                        },
                    ]
                    msgs2send.append(update_msg)
            total_cost = 0
            if all_messages:
                total_cost = all_messages[0].total_cost or 0
            openai_client = get_openai_client()
            response = openai_client.chat.completions.create(
                model=self.sel_thread.model,
                messages=msgs2send
            )
            usage = {}
            if response.usage:
                usage = {
                    "completion_tokens": float(response.usage.completion_tokens),
                    "prompt_tokens": float(response.usage.prompt_tokens),
                    "total_tokens": float(response.usage.total_tokens),
                }
                pricing = self.model_pricings.get(self.sel_thread.model) or {}
                usage['cost'] = usage['completion_tokens'] * float(pricing.get('output_tokens', 0))
                usage['cost'] += usage['prompt_tokens'] * float(pricing.get('input_tokens', 0))
            total_cost += usage.get('cost') or 0
            Messages.create(thread_id=self.sel_thread.id, role='assistant', message=response.choices[0].message.content,
                            total_cost=total_cost, **usage)
            msgs = self.get_messages(self.sel_thread.id, self.history_size)
            return msgs, total_cost, usage['cost'] or 0
        return None, None, None

    def get_sidebar(self):
        sel_thread_name = st.selectbox(label='Wybierz wątek', options=self.thrs_data.keys())
        self.sel_thread = self.thrs_data[sel_thread_name] if sel_thread_name else None
        # mstct = st.multiselect('Wybierz wiele opcji', ['Opcja 1', 'Opcja 2', 'Opcja 3'])
        self.history_size = st.number_input(label='Wielkość historii', value=5)
        if self.sel_thread:
            self.get_messages(self.sel_thread.id, self.history_size)
            c0, c1 = st.columns([9, 1])
            with c0:
                st.metric("Koszt rozmowy", f"${self.session_state.cache_total:.4f}")
            with c1:
                st.metric("Koszt ostatniego pytania", f"${self.session_state.cache_cost:.4f}")
        with st.expander("Zarządzanie wątkami"):
            with st.form('thread_form', clear_on_submit=True):
                new_thread_name = st.text_input(label='Nazwa nowego wątku')
                sel_chat_idx = self.chat_opts.index(self.sel_thread.model) if self.sel_thread else self.chat_opts.index('gpt-4o-mini')
                chat_model = st.selectbox(label='Wybierz model', options=self.chat_opts, index=sel_chat_idx)
                pers_init = self.all_pers_idxs.get(self.sel_thread.personality_id, 0) if self.sel_thread else None
                personality_val = st.selectbox(label='Wybierz osobowość', options=self.pers_names.keys(), index=pers_init)
                st.form_submit_button('Utwórz wątek')
                if new_thread_name:
                    Threads.create(name=new_thread_name, model=chat_model, personality_id=self.pers_names[personality_val][0])

        if personality_val:
            with st.expander('Edycja osobowości "%s"' % personality_val):
                with st.form('edit_personality_form'):
                    edit_p_name = st.text_input(label='Nazwa osobowiści', value=personality_val)
                    edit_p_value = st.text_area('Opis osobowości', value=self.pers_names[personality_val][1])
                    st.form_submit_button('Edytuj osobowość')
                    if edit_p_name and edit_p_value:
                        if edit_p_name != personality_val or edit_p_value != self.pers_names[personality_val][1]:
                            Personality.objects().filter(id=self.pers_names[personality_val][0]).update(
                                name=edit_p_name, value=edit_p_value
                            )
                st.button('Usuń', on_click=self.remove_personality, args=[self.pers_names[personality_val][0]])
        with st.expander("Nowa osobowość?"):
            with st.form('personality_form', clear_on_submit=True):
                # Jesteś asystentem i odpowiadasz na ogólne pytania
                personality_name = st.text_input(label='Nazwa osobowiści')
                personality_value = st.text_area('Opis osobowości', value='')
                st.form_submit_button('Utwórz osobowość')
            if personality_name and personality_value:
                Personality.create(name=personality_name, value=personality_value)
                self.session_state.num = self.session_state.num + 1

    def get_main_content(self):

        if self.session_state.get('num') is None:
            self.session_state.num = 1

        st.title('Custom Chat GPT')
        prompt = st.chat_input('wprowadź zapytanie')
        upld_file = st.file_uploader(label='Plik', help='Opcjonalny plik', key='upld_file',
                                     type=['png', 'jpg', 'gif', 'webp', 'mp3', 'm4a', 'flac', 'wma', 'aac'])

        if upld_file:
            file_name = upld_file.name
            bts = BytesIO(upld_file.getvalue())
            self.session_state.upload_file = [file_name, bts]

        if self.session_state.get('upload_file'):
            do_upload = st.checkbox(label='Upload %s' % self.session_state.upload_file[0])
        else:
            do_upload = False
        with st.container():
            for msg in self.session_state.messages:
                with st.chat_message(msg['role']):
                    st.write('*(' + smart_date(msg['timestamp']) + ')*')
                    st.write(msg['message'])

        if prompt:
            file_kws = {}
            if do_upload:
                file_kws = {'file': self.session_state.upload_file}
                self.session_state.upload_file = None

            msgs, total, cost = self.add_message(prompt, 'user', **file_kws)
            prompt = ''
            if msgs is not None:
                self.session_state.cache_total = total
                self.session_state.cache_cost = cost
            st.rerun()
