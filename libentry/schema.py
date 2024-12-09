#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "SchemaField",
    "Schema",
    "signature_to_model",
    "parse_type",
]

import enum
from inspect import signature
from typing import Any, Iterable, List, Literal, Mapping, NoReturn, Sequence, Type, Union, get_args, \
    get_origin

from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined


class SchemaField(BaseModel):
    name: str = Field()
    type: Union[str, List[str]] = Field("Any")
    default: Any = Field()
    is_required: bool = Field(True)


class Schema(BaseModel):
    name: str = Field()
    fields: List[SchemaField] = Field(default_factory=list)


def signature_to_model(fn) -> Type[BaseModel]:
    sig = signature(fn)

    fields = {}
    for name, param in sig.parameters.items():
        if name in ["self", "cls"]:
            continue

        annotation = param.annotation
        if annotation is sig.empty:
            annotation = Any

        default = param.default
        field = Field() if default is sig.empty else Field(default)
        fields[name] = (annotation, field)

    return_annotation = sig.return_annotation
    if return_annotation is not NoReturn:
        if return_annotation is sig.empty:
            return_annotation = Any
        fields["return"] = (return_annotation, None)
    return create_model(f"__{fn.__name__}_signature", **fields)


def parse_type(annotation, context: dict) -> Union[str, List[str]]:
    origin = get_origin(annotation)
    if origin is None:
        origin = annotation

    if isinstance(origin, type):
        if origin in {int, float, str, bool}:
            return origin.__name__
        elif issubclass(origin, Mapping):
            dict_args = get_args(annotation)
            if dict_args:
                key_type = dict_args[0]
                if key_type is not str:
                    raise TypeError("Only \"str\" can be used as the type of dict keys.")
                key_type = "str"
                value_type = parse_type(dict_args[1], context)
                if isinstance(value_type, list):
                    raise TypeError("\"Union\" cannot be used as the type of list elements.")
                return f"Dict[{key_type},{value_type}]"
            else:
                return "Dict"
        elif issubclass(origin, Sequence):
            list_args = get_args(annotation)
            if list_args:
                if len(list_args) > 1:
                    raise TypeError("Only ONE type can be used as the type of list elements.")
                elem_type = parse_type(list_args[0], context)
                if isinstance(elem_type, list):
                    raise TypeError("\"Union\" cannot be used as the type of list elements.")
                return f"List[{elem_type}]"
            else:
                return "List"
        elif issubclass(origin, enum.Enum):
            return f"Enum[{','.join(e.name for e in origin)}]"
        elif issubclass(origin, BaseModel):
            _module = origin.__module__
            _name = origin.__name__
            model_name = _name if (_module is None) else f"{_module}.{_name}"

            if (model_name not in context) and (origin is not BaseModel):
                schema = Schema(name=model_name)
                fields = origin.model_fields
                assert isinstance(fields, Mapping)
                for name, field in fields.items():
                    try:
                        schema.fields.append(SchemaField(
                            name=name,
                            type=parse_type(field.annotation, context),
                            default=field.default if field.default is not PydanticUndefined else None,
                            is_required=field.is_required()
                        ))
                    except TypeError as e:
                        raise TypeError(f"{name}: {str(e)}")
                context[model_name] = schema

            return model_name
        elif issubclass(origin, Iterable) and origin.__name__ in {"Iterable", "Generator", "range"}:
            iter_args = get_args(annotation)
            if len(iter_args) != 1:
                raise TypeError("Only ONE type can be used as the type of iterable elements.")
            iter_type = parse_type(iter_args[0], context)
            if isinstance(iter_type, list):
                raise TypeError("\"Union\" cannot be used as the type of iterable elements.")
            return f"Iter[{iter_type}]"
        else:
            _module = origin.__module__
            _name = origin.__name__
            model_name = _name if (_module is None) else f"{_module}.{_name}"
            if model_name == "builtins.NoneType":
                return "NoneType"
            raise TypeError(f"Unsupported type \"{origin}\".")
    else:
        if origin is Any:
            return "Any"
        elif origin is Union:
            return [
                parse_type(arg, context)
                for arg in get_args(annotation)
            ]
        elif origin is Literal:
            enum_args = get_args(annotation)
            return f"Enum[{','.join(map(str, enum_args))}]"
        else:
            raise TypeError(f"Unsupported type \"{origin}\".")
