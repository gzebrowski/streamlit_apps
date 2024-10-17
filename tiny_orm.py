import copy
import datetime
import re
from typing import Type, Union

from pydantic import BaseModel, Field


class MultipleObjectsReturned(Exception):
    pass


class ObjectDoesNotExist(Exception):
    pass


_registered_models: dict[str, Type['AbstractModel']] = {}


def _get_db_type(annot):
    types_map = {int: 'INTEGER', str: 'TEXT', datetime.date: 'TEXT', datetime.datetime: 'TEXT', float: 'REAL'}
    if annot in types_map:
        return types_map[annot], False

    is_nullable = False
    if getattr(annot, '__origin__', None) == Union:
        args = annot.__args__
        if type(None) in args:
            is_nullable = True
        for tp in types_map.keys():
            if tp in args:
                return types_map[tp], is_nullable
    return 'TEXT', is_nullable


def register_model(cls: Type['AbstractModel']):
    assert AbstractModel in getattr(cls, 'mro', lambda: [])()
    _registered_models[cls.__name__.lower()] = cls
    return cls


def esc_sql_v(value, _nested=False) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return 'NULL'
    elif value == (False, None):  # special case
        return ''
    elif isinstance(value, bool):
        return str(int(value))
    elif isinstance(value, datetime.date):
        return "'%s'" % value.isoformat()
    elif isinstance(value, AbstractModel):
        return value.pk or 'NULL'
    elif isinstance(value, (list, tuple)):
        if _nested:
            raise ValueError('Values cannot be nested')
        value = [esc_sql_v(v, _nested=True) for v in value]
        return ' (%s) ' % ','.join(value)
    value = "'%s'" % str(value).replace('\\', r'\\').replace("'", r"\'").replace("?", r"\?")
    return value


def check_is_key(key: str) -> str:
    result = re.match('^[a-zA-Z]+[a-zA-Z0-9_]*$', key)
    assert result
    return key


def order_key(key: str) -> str:
    if key.startswith('-'):
        return check_is_key(key[1:]) + ' DESC'
    return check_is_key(key)


class Q:
    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs
        self._field = args[0] if args and isinstance(args[0], str) else None
        self._expr = ''
        if kwargs:
            items = []
            for k, v in kwargs.items():
                items.append('%s=%s' % (check_is_key(k), esc_sql_v(v)))
            self._expr = '(%s)' % ' AND '.join(items)

    def __ror__(self, value: 'Q'):
        assert isinstance(value, Q)
        self._expr = ' (%s OR %s) ' % (self._expr, value._expr)

    def __or__(self, value: 'Q'):
        assert isinstance(value, Q)
        result = Q()
        result._expr = ' (%s OR %s) ' % (self._expr, value._expr)
        return result

    def __rand__(self, value: 'Q'):
        assert isinstance(value, Q)
        self._expr = ' (%s AND %s) ' % (self._expr, value._expr)

    def __and__(self, value: 'Q'):
        assert isinstance(value, Q)
        result = Q()
        result._expr = ' (%s AND %s) ' % (self._expr, value._expr)
        return result

    def __str__(self):
        return self._expr


class QuerySet:
    def __init__(self, model, conn, _filters=None, _excludes=None, _orders=None, _joins=None, _raw_filters=None):
        self._cursor = None
        self._conn = conn
        self._model = model
        self._filters = _filters or {}
        self._rawfilters = _raw_filters or ''
        self._excludes = _excludes or {}
        self._orders = _orders or []
        self._joins = _joins or []
        self._limit = None
        self._offset = None
        self._dataset = None
        self._curr_pos = 0
        self._total_rows = None

    def _get_cursor(self):
        if self._cursor is None:
            self._cursor = self._conn.cursor()
        return self._cursor

    @classmethod
    def _join_and(cls, fltrs: dict) -> str:
        def check_ends_with(v: str):
            parts = [' LIKE', ' IS', ' NOT', '>', '<', '=']
            return sum([1 if v.endswith(x) else 0 for x in parts]) > 0

        return ' AND '.join(['%s%s%s' % (
            k, ' ' if check_ends_with(k) else '=', esc_sql_v(v)) for k, v in fltrs.items()])

    def _invoke_filters(self) -> str:
        the_filters = self._join_and(self._filters)
        excludes = self._join_and(self._excludes)
        excludes = ('NOT (%s)' % excludes) if excludes else ''
        return ' AND '.join(list(filter(None, [the_filters, str(self._rawfilters), excludes])))

    def _build_query(self) -> str:
        table_name = self._model.__name__.lower()
        query = f'SELECT {table_name}.* FROM {table_name}'
        if self._joins:
            query += ' '.join(self._joins)
        if self._filters or self._excludes or self._rawfilters:
            query += ' WHERE %s ' % self._invoke_filters()
        if self._orders:
            query += ' ORDER BY %s' % ' '.join(self._orders)
        if self._limit:
            query += ' LIMIT %s' % self._limit
            if self._offset:
                query += ' OFFSET %s' % self._offset
        return query

    def _invoke(self):
        query = self._build_query()
        cursor = self._get_cursor()
        cursor.execute(query)
        rows = [dict(x) for x in cursor.fetchall()]
        self._dataset = rows

    def delete(self):
        query = 'DELETE FROM %s' % self._model.__name__.lower()
        if self._filters or self._excludes:
            query += ' WHERE %s ' % self._invoke_filters()
        cursor = self._get_cursor()
        cursor.execute(query)
        self._conn.commit()

    def update(self, **kwargs):
        values = []

        def gather_val(v2):
            values.append(v2)
            return '?'

        query = 'UPDATE %s SET ' % self._model.__name__.lower()
        query += (','.join(['%s=%s' % (check_is_key(k), gather_val(v)) for k, v in kwargs.items()]))
        if self._filters or self._excludes:
            query += ' WHERE %s ' % self._invoke_filters()
        cursor = self._get_cursor()
        cursor.execute(query, values)
        self._conn.commit()

    @property
    def query(self):
        return self._build_query()

    def raw(self, query: str):
        cursor = self._get_cursor()
        cursor.execute(query)
        if query.lower().strip().startswith(('update', 'delete', 'insert')):
            self._conn.commit()
        else:
            return cursor.fetchall()

    def filter(self, *args, **kwargs):
        return self._common_filter(args, kwargs, _type='_filters')

    def _common_filter(self, args, kwargs, _type='_filters'):
        if args:
            assert not [
                1 for arg in args if not isinstance(arg, Q)], 'You can run filters only with kwargs or Q unstances'
            self._rawfilters = ' AND '.join(
                ([str(self._rawfilters)] if self._rawfilters else []) + [str(x) for x in args])
        dct = getattr(self, _type)
        for k, v in kwargs.items():
            k = check_is_key(k)
            if '__' in k:
                parts = k.split('__')
                curr_model = self._model
                cnt = len(parts)
                curr_key = None
                for nr, part in enumerate(parts):
                    curr_key = curr_key or part
                    fld = curr_model.get_model_fields().get(part)
                    part_with_id = f'{part}_id'
                    fld_rel = curr_model.get_model_fields().get(part_with_id)
                    fld = fld or fld_rel
                    part2 = part_with_id if fld_rel else part
                    if fld:
                        if part == 'pk':
                            part = 'id'
                        field_opts = AbstractModel.get_field_opts(fld)
                        if field_opts.get('relation') and fld_rel and nr < cnt - 1:
                            rel_tbl, _ = self._model.get_rel_model(part2)
                            join_clause = self._get_join(curr_model, part2)
                            if join_clause not in self._joins:
                                self._joins.append(join_clause)
                            curr_model = AbstractModel.get_model(rel_tbl)
                            curr_key = part
                        else:
                            curr_key = f'{curr_model.__name__.lower()}.{part}'
                            if nr == cnt - 1:
                                dct[curr_key] = v
                    else:
                        if part in ['isnull', 'notin', 'in', 'contains', 'startswith', 'endswith', 'gt', 'lt', 'gte',
                                    'lte'] and nr == cnt - 1:
                            if part == 'isnull':
                                k = curr_key + (' IS' if v else ' IS NOT')
                                v = None
                            elif part in ('in', 'notin'):
                                k = curr_key + (' IN ' if part == 'in' else ' NOT IN ')
                            elif part in ('gt', 'lt', 'gte', 'lte'):
                                k = f'{curr_key} %s' % {'gt': '>', 'lt': '<', 'gte': '>=', 'lte': '<='}[part]
                            else:
                                k = f'{curr_key} LIKE'
                                fmt = {'contains': '%%%s%%', 'startswith': '%s%%', 'endswith': '%%%s'}.get(part)
                                v = fmt % str(v)
                            dct[k] = v
                            break
                        else:
                            raise ValueError(f'Invalid lookup {part}')
                # dct[curr_key] = v
            else:
                k = 'id' if k == 'pk' else k
                dct[k] = v
        return self

    def exclude(self, *args, **kwargs):
        return self._common_filter(args, kwargs, _type='_excludes')

    def all(self):
        return self.__class__(self._model, self._conn, _filters=self._filters, _excludes=self._excludes,
                              _orders=self._orders, _joins=self._joins, _raw_filters=self._raw_filters)

    def first(self):
        self._limit = 1
        self._invoke()
        return self._model._prepare_object(self._dataset[0]) if self._dataset else None

    def exists(self):
        return bool(self.first())

    @classmethod
    def _get_join(cls, model, field):
        tbl, rel_f = model.get_rel_model(field)
        mod_name = model.__name__.lower()
        return f' INNER JOIN {tbl} ON {mod_name}.{field}={tbl}.{rel_f} '

    def select_related(self, field):
        join_clause = self._get_join(self._model, field)
        if join_clause not in self._joins:
            self._joins.append(join_clause)
        return self

    def count(self):
        self._invoke()
        return len(self._dataset)

    def get(self, pk=None, **kwargs):
        if pk:
            self._filters['id'] = pk
        if kwargs:
            self.filter(**kwargs)
        self._limit = 2
        self._invoke()
        if len(self._dataset) > 1:
            raise MultipleObjectsReturned
        elif not self._dataset:
            raise ObjectDoesNotExist
        return self._model._prepare_object(self._dataset[0])

    def order_by(self, *args):
        self._orders = [order_key(k) for k in args]
        return self

    def __getitem__(self, val):
        if isinstance(val, int):
            assert val >= 0
            self._offset = val
            self._limit = 1
        else:
            self._offset = val.start or 0
            if val.stop and val.stop > self._offset:
                self._limit = (val.stop - self._offset)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._dataset is None:
            self._curr_pos = 0
            self._invoke()
            self._total_rows = len(self._dataset)
        if self._curr_pos >= self._total_rows:
            self._curr_pos = 0
            raise StopIteration
        dt = copy.deepcopy(self._dataset[self._curr_pos])
        id_ = dt.pop('id', None)
        result = self._model(**dt)
        result._id = id_
        self._curr_pos += 1
        return result


class AbstractModel(BaseModel):
    @staticmethod
    def get_field_opts(field):
        schema_opts = field.json_schema_extra or {}
        schema_opts = {k.lower().strip(): v for k, v in schema_opts.items()}
        return schema_opts

    @classmethod
    def _prepare_object(cls, data):
        id_ = data.pop('id', None)
        result = cls(**data)
        result._id = id_
        return result

    @classmethod
    def create_table(cls):
        table_name = (cls.__name__).lower()
        lines = []
        constrains = []
        primary_exists = False
        for f_name, f in cls.model_fields.items():
            schema_opts = cls.get_field_opts(f)
            tp, is_nullable = _get_db_type(f.annotation)
            extra = [] if is_nullable else ['NOT NULL']
            if schema_opts.get('primary'):
                extra.append('PRIMARY KEY')
                primary_exists = True
            elif schema_opts.get('default'):
                if not callable(schema_opts['default']):
                    extra.append('DEFAULT %s' % esc_sql_v(schema_opts['default']))
            elif schema_opts.get('unique'):
                extra.append('UNIQUE')
            elif schema_opts.get('relation'):
                rel_tbl, rel_col = cls.get_rel_model(f_name)
                extra_rel_opts = schema_opts.get('relation_ots') or ''
                constrains.append('FOREIGN KEY (%s) REFERENCES %s (%s) %s' % (f_name, rel_tbl, rel_col, extra_rel_opts))
            line = f'{f_name} {tp} ' + ' '.join(extra)
            lines.append(line)
        if not primary_exists:
            lines = ['id INTEGER NOT NULL PRIMARY KEY'] + lines
        sql_create = 'CREATE TABLE IF NOT EXISTS %s (%s)' % (table_name, ','.join(lines + constrains))
        conn = cls._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql_create)

    @property
    def pk(self):
        return getattr(self, '_id', None)

    @property
    def id(self):
        return getattr(self, '_id', None)

    def __getattr__(self, attr):
        field_name = f'{attr}_id'
        fld = self.__class__.model_fields.get(field_name)
        if fld and self.get_field_opts(fld).get('relation'):
            pk = getattr(self, field_name)
            if pk:
                rel_model = self.get_model(self.get_rel_model(field_name)[0])
                return rel_model.objects().get(pk)
        return None

    @classmethod
    def _get_connection(cls):
        raise NotImplementedError

    @classmethod
    def objects(cls, connection=None):
        conn2 = connection or cls._get_connection()
        return QuerySet(cls, conn2)

    @classmethod
    def get_rel_model(cls, field_name: str):
        opts = cls.get_field_opts(cls.model_fields[field_name])
        rel = opts['relation']
        return (rel[0].lower(), rel[2]) if isinstance(rel, (list, tuple)) else (rel.lower(), 'id')

    @staticmethod
    def get_model(name: str) -> Type['AbstractModel']:
        return _registered_models[name]

    @classmethod
    def get_model_fields(cls):
        fields = dict(cls.model_fields)
        fields.update({'id': Field(), 'pk': Field()})
        return fields

    @classmethod
    def create(cls, **kwargs):
        connection = kwargs.pop('connection', None) or cls._get_connection()
        items = list(kwargs.items())
        columns = [check_is_key(x[0]) for x in items]
        # values = [esc_sql_v(x[1]) for x in items]
        values = [x[1] for x in items]
        for f_name, mod_col in cls.model_fields.items():
            opts = cls.get_field_opts(mod_col)
            if f_name not in columns:
                add = False
                v2 = None
                if opts.get('default'):
                    v2 = opts['default']() if callable(opts['default']) else opts['default']
                    add = True
                if opts.get('auto_now_add') and mod_col.annotation in (datetime.date, datetime.datetime):
                    v2 = datetime.date.today() if mod_col.annotation == datetime.date else datetime.datetime.now()
                    add = True
                if add:
                    columns.append(f_name)
                    # values.append(esc_sql_v(v2))
                    values.append(v2)
        sql = f'INSERT INTO {cls.__name__} (%s) VALUES (%s)' % (
            ','.join(columns), ','.join(['?'] * len(values)))
        cursor = connection.cursor()
        cursor.execute(sql, values)
        connection.commit()
        return cursor.lastrowid

    def delete(self):
        self.objects().filter(id=self.id).delete()

    def save(self):
        fields = self.__class__.model_fields.keys()
        dt = {k: getattr(self, k) for k in fields}
        self.objects().filter(id=self.id).update(**dt)
        return self


def prepare_db():
    for md in _registered_models.values():
        md.create_table()
