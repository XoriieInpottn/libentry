#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "SchemaField",
    "Schema",
    "ParseContext",
    "parse_type",
    "QueryAPIOutput",
    "query_api",
]

import enum
from dataclasses import asdict, dataclass, is_dataclass
from inspect import signature
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, NoReturn, Optional, Sequence, Union, \
    get_args, \
    get_origin

from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined


class SchemaField(BaseModel):
    name: str = Field()
    type: Union[str, List[str]] = Field("Any")
    default: Any = Field()
    is_required: bool = Field(True)
    title: str = Field()
    description: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Schema(BaseModel):
    name: str = Field()
    fields: List[SchemaField] = Field(default_factory=list)


@dataclass
class ParseContext:
    annotation: Any
    schemas: MutableMapping[str, Schema]
    origin: Any


_TYPE_PARSERS = []
_GENERIC_PARSERS = []


def parse_type(annotation, context: MutableMapping[str, Schema]) -> Union[str, List[str]]:
    origin = get_origin(annotation)
    if origin is None:
        origin = annotation
    pc = ParseContext(
        annotation=annotation,
        schemas=context,
        origin=origin
    )

    parser_list = _TYPE_PARSERS if isinstance(origin, type) else _GENERIC_PARSERS
    for parser in parser_list:
        output = parser(pc)
        if output is not None:
            return output
    raise TypeError(f"Unsupported type \"{origin}\".")


def type_parser(fn):
    _TYPE_PARSERS.append(fn)
    return fn


@type_parser
def _parse_basic_types(context: ParseContext):
    if context.origin in {int, float, str, bool}:
        return context.origin.__name__


@type_parser
def _parse_dict(context: ParseContext):
    if issubclass(context.origin, Mapping):
        dict_args = get_args(context.annotation)
        if dict_args:
            key_type = dict_args[0]
            if key_type is not str:
                raise TypeError("Only \"str\" can be used as the type of dict keys.")
            key_type = "str"
            value_type = parse_type(dict_args[1], context.schemas)
            if isinstance(value_type, list):
                raise TypeError("\"Union\" cannot be used as the type of dict elements.")
            return f"Dict[{key_type},{value_type}]"
        else:
            return "Dict"


@type_parser
def _parse_list(context: ParseContext):
    if issubclass(context.origin, Sequence):
        list_args = get_args(context.annotation)
        if list_args:
            if len(list_args) > 1:
                raise TypeError("Only ONE type can be used as the type of list elements.")
            elem_type = parse_type(list_args[0], context.schemas)
            if isinstance(elem_type, list):
                raise TypeError("\"Union\" cannot be used as the type of list elements.")
            return f"List[{elem_type}]"
        else:
            return "List"


@type_parser
def _parse_enum(context: ParseContext):
    if issubclass(context.origin, enum.Enum):
        return f"Enum[{','.join(e.name for e in context.origin)}]"


@type_parser
def _parse_base_model(context: ParseContext):
    origin = context.origin
    if issubclass(origin, BaseModel):
        _module = origin.__module__
        _name = origin.__name__
        model_name = _name if (_module is None) else f"{_module}.{_name}"

        is_new_model = model_name not in context.schemas
        is_not_base_class = origin is not BaseModel
        if is_new_model and is_not_base_class:
            schema = Schema(name=model_name)
            fields = origin.model_fields
            assert isinstance(fields, Mapping)
            for name, field in fields.items():
                try:
                    default = field.default if field.default is not PydanticUndefined else None
                    title = field.title
                    if title is None:
                        title = "".join(word.capitalize() for word in name.split("_"))
                    schema_field = SchemaField(
                        name=name,
                        type=parse_type(field.annotation, context.schemas),
                        default=default,
                        is_required=field.is_required(),
                        title=title,
                        description=field.description
                    )
                except TypeError as e:
                    raise TypeError(f"{name}: {str(e)}")
                for md in field.metadata:
                    if is_dataclass(md):
                        schema_field.metadata.update(asdict(md))
                schema.fields.append(schema_field)
            context.schemas[model_name] = schema

        return model_name


@type_parser
def _parse_iterable(context: ParseContext):
    if context.origin.__name__ in {"Iterable", "Generator", "range"} and issubclass(context.origin, Iterable):
        iter_args = get_args(context.annotation)
        if len(iter_args) != 1:
            raise TypeError("Only ONE type can be used as the type of iterable elements.")
        iter_type = parse_type(iter_args[0], context.schemas)
        if isinstance(iter_type, list):
            raise TypeError("\"Union\" cannot be used as the type of iterable elements.")
        return f"Iter[{iter_type}]"


@type_parser
def _parse_none_type(context: ParseContext):
    origin = context.origin
    if origin.__module__ == "builtins" and origin.__name__ == "NoneType":
        return "NoneType"


@type_parser
def _parse_ndarray(context: ParseContext):
    origin = context.origin
    if origin.__module__ == "numpy" and origin.__name__ == "ndarray":
        return "numpy.ndarray"


def generic_parser(fn):
    _GENERIC_PARSERS.append(fn)
    return fn


@generic_parser
def _parse_any(context: ParseContext):
    if context.origin is Any or str(context.origin) == str(Any):
        return "Any"


@generic_parser
def _parse_union(context: ParseContext):
    if context.origin is Union or str(context.origin) == str(Union):
        return [
            parse_type(arg, context.schemas)
            for arg in get_args(context.annotation)
        ]


@generic_parser
def _parse_literal(context: ParseContext):
    if context.origin is Literal or str(context.origin) == str(Literal):
        enum_args = get_args(context.annotation)
        return f"Enum[{','.join(map(str, enum_args))}]"


class QueryAPIOutput(BaseModel):
    input_schema: str
    output_schema: str
    context: Mapping[str, Schema]
    bundled_input: bool


def query_api(fn) -> QueryAPIOutput:
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

    args_model = None
    if len(fields) == 1:
        for annotation, _ in fields.values():
            origin = get_origin(annotation)
            if origin is None:
                origin = annotation
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                args_model = origin
    bundle = args_model is None
    if bundle:
        name = "".join(word.capitalize() for word in fn.__name__.split("_"))
        args_model = create_model(f"{name}Request*", **fields)

    context = {}
    input_schema = parse_type(args_model, context)
    output_schema = None
    return_annotation = sig.return_annotation
    if return_annotation is not None and return_annotation is not NoReturn:
        if return_annotation is sig.empty:
            return_annotation = Any
        output_schema = parse_type(return_annotation, context)
    if isinstance(output_schema, list):
        output_schema = output_schema[0]

    return QueryAPIOutput(
        input_schema=input_schema,
        output_schema=output_schema,
        context=context,
        bundled_input=bundle,
    )
