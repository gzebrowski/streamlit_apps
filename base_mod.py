from abc import ABC, abstractmethod
from hashlib import sha1
from typing import Any

import streamlit as st
from openai import OpenAI

from streamlit_env import Env

env = Env('.env')

OPENAI_API_KEY = env["OPENAI_API_KEY"]


def get_bool_env(key: str):
    val = env.get(key)
    if val:
        if str(val).lower().strip() in ['0', 'false', 'no', '']:
            return False
    return bool(val)


class ProxySessionState:
    def __init__(self, spacename):
        self._spacename = spacename
        if spacename not in st.session_state:
            st.session_state[spacename] = {}
        self._s_st = st.session_state
        self._inited = False

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            return super().__setattr__(key, value)
        state = st.session_state[self._spacename]
        state[key] = value
        st.session_state[self._spacename] = state

    def __getattr__(self, key: str) -> Any:
        return st.session_state[self._spacename].get(key)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.__setattr__(key, value)

    @property
    def _is_inited(self):
        if self._inited:
            return True
        if st.session_state[self._spacename].get('inited_'):
            return True
        return False

    def _do_init(self, items):
        for k, v in items:
            self.__setattr__(k, v)
        self._set_inited()

        self._set_inited()

    def _set_inited(self):
        dt = st.session_state[self._spacename]
        dt['inited_'] = True
        st.session_state[self._spacename] = dt
        self._inited = True

    def get(self, key: str, default: Any = None):
        result = self.__getattr__(key)
        return default if result is None else result


class BaseStreamlitApp(ABC):
    SHOW_SIDEBAR = False
    PRIORITY = 0
    SESSION_DEFAULTS = []
    NAME = ''

    def get_sidebar(self):
        pass

    @abstractmethod
    def get_main_content(self):
        pass

    def __init__(self):
        name = self.__class__.__name__
        self.session_state = ProxySessionState(name)
        if not self.session_state._is_inited:
            init_items = list(self.SESSION_DEFAULTS)
            if 'logs_' not in init_items:
                init_items.append(('logs_', []))
            self.session_state._do_init(self.SESSION_DEFAULTS)
        self.init()

    def init(self):
        pass

    def log(self, *args):
        if len(args) == 1:
            line = args[0]
        else:
            line = ' '.join([str(x) for x in args])
        self.session_state.logs_ = self.session_state.get('logs_', []) + [line]

    def get_logs(self):
        return getattr(self.session_state, 'logs_', None) or []

    def clear_logs(self):
        self.session_state.logs_ = []

    def get_api_key(self):
        api_key = self.session_state.api_key
        if not api_key:
            api_key = env.get('OPENAI_API_KEY') if not get_bool_env('LOGIN_MODE') else None
            api_key = api_key or st.text_input('Podaj klucz api22', type='password')
            if api_key:
                hsh = env.get('PWD_HASH')
                salt = env.get('PWD_SALT')
                env_api_key = env.get('OPENAI_API_KEY')
                if hsh and salt and env_api_key:
                    if sha1(f'{salt} {api_key}'.encode()).hexdigest() == hsh:
                        api_key = env_api_key
                self.session_state.api_key = api_key
                st.rerun()
        return api_key


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)
