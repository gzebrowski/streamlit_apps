import json
import math
import random
from pathlib import Path
from typing import Optional

import instructor
import numpy as np
import pandas as pd
import streamlit as st
from audiorecorder import audiorecorder
from faker import Faker
from pycaret.clustering import (assign_model, create_model, load_model,
                                predict_model, save_model)
from pycaret.clustering import setup as setup_model
from pydantic import BaseModel

from base_mod import BaseStreamlitApp, env, get_openai_client
from dynamic_filters import DynamicFilters
from helpers import (get_mp3_audio_and_hash, transcribe_audio, write_html,
                     write_small)
from qdrant_helpers import QdrantHelper

EDU_LEVEL_CHOICES = ['Podstawowe', 'Średnie', 'Wyższe'], [0.13, 0.53, 0.34]
FAV_PLACES_CHOICES = ['nad wodą', 'w lesie', 'w górach', 'inne'], [0.3, 0.15, 0.45, 0.1]
FAV_ACTIVITIES = [
    ('filmy', 'movies'),
    ('muzyka', 'music'),
    ('książki', 'books'),
    ('spacery', 'walking'),
    ('jazda na rowerze', 'cycling'),
    ('kino', 'cinema'),
    ('teatr', 'theathre'),
    ('taniec', 'dance'),
    ('gry komputerowe', 'computer_games'),
    ('sport', 'sport'),
    ('fintess', 'fitness'),
    ('malowanie', 'painting'),
    ('wolontariat', 'voluntary_service'),
    ('fotografia', 'photography'),
    ('inne', 'other'),
]

FAITH_CHOICES = [
    ['katolik praktykujący', 'katolik niepraktykujący', 'protestant / ewangeliczny chrześcijanin',
        'ateista', 'prawosławny', 'inne wyznania'],
    [0.30, 0.50, 0.01, 0.12, 0.01, 0.06]
]
POLITICAL_CONVICTIONS_FAITH_RELATION = [
    ('lewicowy', (0.01, 0.2, 0.01, 0.4, 0.01, 0.01)),
    ('liberalny', (0.24, 0.6, 0.24, 0.59, 0.24, 0.24)),
    ('prawicowy', (0.75, 0.2, 0.75, 0.01, 0.75, 0.75)),
]


def get_age_category(age: Optional[int]) -> str:
    if not age:
        return None
    age_category_tmp = (age - 18) // 5
    return '%s-%s' % (age_category_tmp * 5 + 18, (age_category_tmp + 1) * 5 + 18)


data_to_gather: list[tuple[str, str, dict]] = [
    ('gender', 'podaj swoją płeć', {'prefix': 'Płeć ', 'values': ['mężczyzna', 'kobieta']}),
    ('education_level', 'podaj wykształcenie', {'prefix': 'wykształcenie ', 'values': EDU_LEVEL_CHOICES[0]}),
    ('favorite_place', 'podaj, gdzie najbardziej lubisz wypoczywać', {'prefix': 'Ulubione miejsce wypoczynku ',
                                                                      'values': FAV_PLACES_CHOICES[0]}),
    ('favorite_activity', 'podaj, ulubiony sposób spędzania wolnego czasu',
     {'prefix': 'Ulubiony sposób spędzania wolnego czasu ', 'values': [f[0] for f in FAV_ACTIVITIES]}),
    ('religious_faith', 'podaj wyznanie', {'prefix': 'Wyznanie ', 'values': FAITH_CHOICES[0]}),
    ('political_choice', 'podaj przekonania polityczne',
     {'prefix': 'opcja polityczna ', 'values': [x[0] for x in POLITICAL_CONVICTIONS_FAITH_RELATION]}),
    ('age_category', 'podaj wiek', {'numeric': True, 'format': get_age_category, 'help_text': 'Podaj samą liczbę lat',
                                    '_vals': sorted(list(set([get_age_category(x) for x in range(18, 110, 5)])))}),
]


qdrant_helper = QdrantHelper(get_openai_client())


class VoicePollModule(BaseStreamlitApp):
    SHOW_SIDEBAR = True
    NAME = 'Voice Poll'
    SESSION_DEFAULTS = [('gathered_data', {}), ('audio_hashes', []), ('audio_story_hash', None), ('logs', []),
                        ('confirmed_poll', False)]

    def __init__(self):
        super().__init__()
        self.gathered_data = dict(self.session_state.gathered_data)

        self.replacements = {'age_category': ('age', int)}
        self.rev_replacements = dict([(v[0], k) for k, v in self.replacements.items()])

        self.all_model_fields = [(f[0], str) for f in data_to_gather if f[0] not in self.replacements]
        self.all_model_fields.extend(list(self.replacements.values()))
        self.log('all_model_fields', self.all_model_fields)

        kmeans_pipeline, cluster_names_and_descriptions, search_data, full_data = self.load_my_model()
        self.kmeans_pipeline = kmeans_pipeline
        self.cluster_names_and_descriptions = cluster_names_and_descriptions
        self.search_data = search_data
        self.dyn_f = DynamicFilters(full_data)

        qdrant_helper.index_embedings([
            {
                'name': x[0],
                'values': [x[2]['prefix'] + v for v in x[2]['values']],
            } for x in data_to_gather if x[2].get('values')])

    def get_or_create_fake_corelated_data(self, filename):
        try:
            people = pd.read_csv(filename, sep=';')
        except FileNotFoundError:
            pass
        else:
            return people
        faker = Faker('pl')
        people = []
        favorite_correlations = {
            'movies': {'w': 0.3},
            'music': {'w': 0.15},
            'books': {'w': 0.5, 'edu_level': (0.1, 0.3, 0.6), 'gender': (0.7, 0.3)},
            'walking': {'w': 0.3},
            'cycling': {'w': 1, 'bmi': (0.6, 0.3, 0.1)},
            'cinema': {'w': 0.20},
            'theathre': {'w': 0.3, 'edu_level': (0.1, 0.3, 0.6), 'gender': (0.6, 0.4)},
            'dance': {'w': 2.5, 'edu_level': (0.45, 0.3, 0.25), 'bmi': (0.5, 0.3, 0.2), 'gender': (0.6, 0.4)},
            'computer_games': {'w': 0.8, 'edu_level': (0.6, 0.3, 0.1), 'bmi': (0.2, 0.3, 0.5), 'gender': (0.1, 0.9)},
            'sport': {'w': 0.5, 'bmi': (0.6, 0.3, 0.1)},
            'fitness': {'w': 0.4, 'bmi': (0.6, 0.3, 0.1), 'gender': (0.9, 0.1)},
            'painting': {'w': 0.15, 'gender': (0.7, 0.3)},
            'voluntary_service': {'w': 0.05},
            'photography': {'w': 0.15},
            'other': {'w': 0.2}
        }
        fav_places_n, fav_places_v = FAV_PLACES_CHOICES
        rows_count = int(env.get('ROWS_NUMBER', 2000))
        for _ in range(rows_count):
            sex = 'mężczyzna' if np.random.random(1)[0] > 0.55 else 'kobieta'
            height_params = [165, 6, 1] if sex == 'kobieta' else [180, 7, 1]
            height = int(np.random.normal(*height_params)[0])
            bmi = np.random.normal(26.5, 4.5, 1)[0]
            bmi_level = max(0, min(2, (bmi // 5) - 4))  # should be 0, 1 or 2
            weight = int(bmi * math.pow((height / 100), 2))
            first_name = str(faker.first_name_female() if sex == 'kobieta' else faker.first_name_male())
            edu_level_nr = np.random.choice([0, 1, 2], p=EDU_LEVEL_CHOICES[1])
            edu_level = EDU_LEVEL_CHOICES[0][edu_level_nr]
            sex_nr = 0 if sex == 'kobieta' else 1
            faith_data = FAITH_CHOICES
            political_convictions_faith_relation = POLITICAL_CONVICTIONS_FAITH_RELATION
            religious_faith = np.random.choice(len(faith_data[0]), 1, p=faith_data[1])[0]
            political_convictions_p_param = [x[1][religious_faith] for x in political_convictions_faith_relation]
            political_convictions = np.random.choice(
                len(political_convictions_faith_relation), 1,
                p=political_convictions_p_param)[0]
            item = {
                'age': 18 + int(max(0, np.random.normal(18, 8, 1)[0])),
                'first_name': first_name,
                'last_name': faker.last_name(),
                'gender': sex,
                'height': height,
                'weight': weight,
                'education_level': edu_level,
                'religious_faith': faith_data[0][religious_faith],
                'political_choice': political_convictions_faith_relation[political_convictions][0],
                'favorite_place': np.random.choice(fav_places_n, 1, p=fav_places_v)[0],
            }
            item['age_category'] = get_age_category(int(item['age']))
            factor_levels = {'bmi': bmi_level, 'education_level': edu_level_nr, 'gender': sex_nr}
            fl_keys = list(factor_levels.keys())
            for fav, fav_factors in favorite_correlations.items():
                p = float(fav_factors['w'])
                for fl_key in fl_keys:
                    if fav_factor := fav_factors.get(fl_key):
                        p *= fav_factor[int(factor_levels[fl_key])]
                item[f'favorite_{fav}'] = 1 if random.random() <= p else 0
            people.append(item)
        people = pd.DataFrame(people, index=list(range(1, rows_count + 1)))
        people.to_csv(filename, sep=';', index=None)
        return people

    def generate_cluster_descriptions(self, data_with_kmeans):
        path = Path(env['CLUSTER_DESCRIPTIONS_PATH'])
        if path.exists():
            return json.loads(open(path).read())
        all_clusters = data_with_kmeans['Cluster'].unique()

        data = {}
        for clstr in all_clusters:
            df: pd.DataFrame = data_with_kmeans[data_with_kmeans['Cluster'] == clstr]
            c_k = str(clstr)
            data[c_k] = []
            for column in df.columns:
                if column == 'Cluster':
                    continue
                value_counts = df.value_counts(subset=column)
                data[c_k].append([column, ', '.join([f'{c}: {cnt}' for c, cnt in value_counts.items()])])
        summary = '\n\n'.join([(f'Klaster "{claster}":\n' + '\n'.join([f'{c}: {d}' for c, d in lines]))
                               for claster, lines in data.items()])
        ai_prompt = f'Użyliśmy algorytmu klastrowania\n{summary}\n\n'\
            'Wygeneruj najlepsze nazwy dla każdego z klastrów oraz ich opisy\n\nUżyj formatu JSON. Przykładowo:\n'\
            '''
                {
                    "Cluster 0": {
                        "name": "<krótka nazwa Klastra 0>",
                        "description": "W tej kategorii znajdują się osoby, które..."
                    }
                    "Cluster 1": {
                        "name": "<krótka nazwa Klastra 1>",
                        "description": "W tej kategorii znajdują się osoby, które..."
                    },
                    ...
                }
            '''
        openai_client = get_openai_client()

        response = openai_client.chat.completions.create(model=env['DESC_AI_MODEL'], temperature=0, messages=[{
            'role': 'user',
            'content': [{'type': 'text', 'text': ai_prompt}],
        }])
        result = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
        cluster_names_desc = json.loads(result)
        with open(path, 'w') as f:
            f.write(json.dumps(cluster_names_desc))
        return cluster_names_desc

    def load_my_model(self, try_again=False):
        my_model_name = env['MY_MODEL_NAME']
        path = Path(f'{my_model_name}.pkl')
        people_data_df = self.get_or_create_fake_corelated_data(Path(env['PEOPLE_DATA_FILENAME']))
        columns = ['gender', 'education_level', 'religious_faith', 'political_choice', 'age_category']
        columns += [c for c in people_data_df.columns if c.startswith('favorite_')]
        df = people_data_df[columns]
        setup_model(df, session_id=env.get('MODEL_SESSION_ID', 123))
        if path.exists():
            kmeans_ppln = load_model(my_model_name)
            descs = self.generate_cluster_descriptions(kmeans_ppln)
            data_with_kmeans = assign_model(kmeans_ppln)
            return kmeans_ppln, descs, data_with_kmeans, people_data_df
        if try_again:
            raise RuntimeError
        kmeans = create_model('kmeans', num_clusters=16)
        data_with_kmeans = assign_model(kmeans)
        self.generate_cluster_descriptions(data_with_kmeans)
        save_model(kmeans, my_model_name, verbose=False)
        return self.load_my_model(True)

    def find_cluster(self, kmeans_ppl, data):
        predict_with_cluster_df = predict_model(model=kmeans_ppl, data=data)
        return predict_with_cluster_df['Cluster']

    def retrieve_structure(self, openai_client, text: str, response_model: BaseModel) -> dict:
        instructor_openai_client = instructor.from_openai(openai_client)
        res = instructor_openai_client.chat.completions.create(
            model=env['DESC_AI_MODEL'],
            temperature=0,
            response_model=response_model,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        return res.model_dump()

    def categorize_person(self, provided_data, kmeans_ppln, cluster_names_descriptions):
        fav_activity = provided_data.pop('favorite_activity')
        data_person = {
            k: provided_data[k] for k in provided_data
        }
        for f_pol, f_field in FAV_ACTIVITIES:
            data_person[f'favorite_{f_field}'] = 1 if f_pol == fav_activity[1] else 0

        person_df = pd.DataFrame([data_person])
        predicted_cluster_id = predict_model(kmeans_ppln, data=person_df)["Cluster"].values[0]
        predicted_cluster_data = cluster_names_descriptions[predicted_cluster_id]
        idxs = self.search_data[self.search_data['Cluster'] == predicted_cluster_id].index.to_list()
        return predicted_cluster_id, predicted_cluster_data, idxs

    def get_main_content(self):
        tab1, tab2, tab3 = st.tabs(['Opowiedz o sobie', 'eksploruj dane', 'logs'])

        with tab1:
            cls_fields = {}
            cls_annotations = {}
            anything_provided = False

            for f, tp in self.all_model_fields:
                f2 = self.rev_replacements.get(f, f)
                if f2 not in self.gathered_data:
                    cls_fields[f] = None
                    cls_annotations[f] = Optional[tp]
                else:
                    anything_provided = True

            my_model_fields = cls_fields | {'__annotations__': cls_annotations}
            MyModel = type('MyModel', (BaseModel,), my_model_fields)
            # my_model = MyModel()
            st.header('Opowiedz nam o sobie')
            data_to_gather_dict = dict([(rec[0], rec[1:]) for rec in data_to_gather])
            if not anything_provided:
                st.write(
                    '''podaj swoją płeć, swój wiek, swoje wykształcenie, ulubione miejsce wypoczynku,
                    ulubiony sposób spędzania wolnego czasu, swoje przekonania religijne i polityczne''')
            elif cls_fields:
                st.write('Brakuje nam jeszcze trochę danych. Prosimy nagrać się jeszcze raz i podać następujące dane:')
                not_provided_fields = cls_fields.keys()
                needed_data = [f[0] for f in self.all_model_fields if f[0] in not_provided_fields]
                for f in needed_data:
                    f2 = self.rev_replacements.get(f, f)
                    st.write('- ' + data_to_gather_dict[f2][0])
                    if data_to_gather_dict[f2][1].get('values'):
                        write_small('np. ' + ', '.join(data_to_gather_dict[f2][1]['values']))

            self.log('gathered_data', self.gathered_data)
            if cls_fields:
                rec_key = '_'.join(sorted(list(cls_fields.keys())) + [self.session_state.audio_story_hash or ''])
                curr_rec_story = audiorecorder(start_prompt='Nagraj', stop_prompt='Zakończ', key=rec_key)
                if curr_rec_story:
                    audio_bytes_story, hsh2 = get_mp3_audio_and_hash(curr_rec_story[:60000])
                    if hsh2 != self.session_state.audio_story_hash:
                        self.session_state.audio_story_hash = hsh2
                        with st.spinner('Przetwarzam...'):
                            txt2 = transcribe_audio(get_openai_client(), audio_bytes_story)
                        with st.spinner('Przetwarzam...'):
                            result3 = self.retrieve_structure(get_openai_client(), txt2, MyModel)
                        self.log('data from AI', result3, 'transcribed text:', txt2)
                        for k, v in result3.items():
                            k2 = self.rev_replacements.get(k, k)
                            if v is not None and v != '':
                                opts = data_to_gather_dict[k2][1]
                                if k == 'age':
                                    if str(v).isnumeric():
                                        self.gathered_data[k2] = v, get_age_category(int(v)), None
                                else:
                                    recs = qdrant_helper.search_values_from_db(k2, query=v, limit=2)
                                    self.log('qdrant_search', k2, v, recs)
                                    self.gathered_data[k2] = v, recs[0]['text'][len(opts['prefix']):], recs
                        self.session_state.gathered_data = self.gathered_data
                        st.rerun()
            else:
                gathered_data2 = {k: v[1] for k, v in self.gathered_data.items()}
                poll_itm = {}
                if not self.session_state.confirmed_poll:
                    st.write('Zainicjowaliśmy wstępnie ankietę o Tobie. Prosimy o ew. skorygowanie danych')
                    for itm2 in data_to_gather:
                        k5 = itm2[0]
                        poll_itm_opts = itm2[2].get('values') or itm2[2].get('_vals') or []
                        pl_itm_idx = poll_itm_opts.index(gathered_data2[k5]) if gathered_data2[k5] else None
                        poll_itm[k5] = st.radio(itm2[1], options=poll_itm_opts, index=pl_itm_idx)
                        if poll_itm[k5]:
                            gathered_data2[k5] = poll_itm[k5]
                    confirm_poll = st.button('Potwierdź dane', key='confirm_poll_btn')
                    if confirm_poll:
                        self.session_state.confirmed_poll = True
                        st.rerun()
                else:
                    _, pred_cluster_data, ids = self.categorize_person(
                        gathered_data2, self.kmeans_pipeline, self.cluster_names_and_descriptions)
                    st.header('Rezultat')
                    write_html(f"<p><strong>Najbardziej dopasowana kategoria:</strong> {pred_cluster_data['name']}</p>")
                    write_html(f"<p>{pred_cluster_data['description']}</p>")
                    st.header('Lista osób z tej kategorii')
                    st.write(self.full_data[self.full_data.index.isin(ids)])

        with tab2:
            self.dyn_f.show_dataframe()

        with tab3:
            for log_line in self.session_state.get('logs') or []:
                st.write(log_line)

    def get_sidebar(self):
        st.header('Eksplorowanie danych')
        self.dyn_f.show_filters()
