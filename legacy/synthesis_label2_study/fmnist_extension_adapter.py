"""Bind frozen functions to an isolated module; keep old suites unchanged."""
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType


def clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    return result


def functions(source, namespace):
    """Copy module exports and rebind its functions, context managers and caches."""
    members = {k: v for k, v in vars(source).items() if not k.startswith('__')}
    namespace.update(members)
    for name, value in members.items():
        raw = getattr(value, '__wrapped__', value)
        if not isinstance(raw, FunctionType) or raw.__module__ != source.__name__:
            continue
        result = clone(raw, namespace)
        if hasattr(value, 'cache_parameters'):
            result = lru_cache(**value.cache_parameters())(result)
        elif raw is not value:
            result = contextmanager(result)
        namespace[name] = result
