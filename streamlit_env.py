import streamlit as st
from dotenv import dotenv_values


class Env:
    def __init__(self, filename):
        self._st_env = dotenv_values(filename)
        self._develop_mode = self._st_env.get('DEVELOP_MODE')

    def _get_env_data(self):
        return self._st_env if self._develop_mode else st.secrets

    def __getitem__(self, key: str) -> str | None:
        _env = self._get_env_data()
        if key not in _env:
            raise IndexError
        return _env.get(key)

    def get(self, key: str, default=None) -> str | None:
        _env = self._get_env_data()
        if key not in _env:
            return default
        return _env.get(key)
