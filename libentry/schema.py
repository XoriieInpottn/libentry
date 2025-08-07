#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APISignature",
    "get_api_signature",
    "SchemaField",
    "Schema",
    "parse_type",
    "QueryAPIOutput",
    "query_api",
]

import enum
from dataclasses import asdict, dataclass, is_dataclass
from inspect import signature
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, NoReturn, Optional, Sequence, Type, \
    Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model
from pydantic_core import PydanticUndefined


class APISignature(BaseModel):
    input_types: List[Any]
    input_model: Optional[Type[BaseModel]] = None
    bundled_model: Optional[Type[BaseModel]] = None
    output_type: Optional[Any] = None
    output_model: Optional[Type[BaseModel]] = None


def get_api_signature(fn, ignores: List[str] = ("self", "cls")) -> APISignature:
    sig = signature(fn)

    input_types = []
    fields = {}
    for name, param in sig.parameters.items():
        if name in ignores:
            continue

        annotation = param.annotation
        if annotation is sig.empty:
            annotation = Any

        input_types.append(annotation)

        default = param.default
        field = Field() if default is sig.empty else Field(default=default)
        fields[name] = (annotation, field)

    input_model = None
    bundled_model = None
    if len(input_types) == 1:
        for annotation in input_types:
            origin = get_origin(annotation) or annotation
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                input_model = origin
    if input_model is None:
        name = "".join(word.capitalize() for word in fn.__name__.split("_"))
        bundled_model = create_model(
            f"{name}Request*",
            __config__=ConfigDict(extra="forbid"),
            **fields
        )

    output_type = None
    output_model = None
    output_annotation = sig.return_annotation
    if output_annotation is not None and output_annotation is not NoReturn:
        if output_annotation is sig.empty:
            output_annotation = Any
        output_type = output_annotation
        output_model = RootModel[output_annotation]

    return APISignature(
        input_types=input_types,
        input_model=input_model,
        bundled_model=bundled_model,
        output_type=output_type,
        output_model=output_model
    )


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

    for parser in _GENERIC_PARSERS:
        output = parser(pc)
        if output is not None:
            return output
    if isinstance(origin, type):
        for parser in _TYPE_PARSERS:
            output = parser(pc)
            if output is not None:
                return output
    raise TypeError(f"Unsupported type \"{origin}\".")


def type_parser(fn):
    _TYPE_PARSERS.append(fn)
    return fn


@type_parser
def _parse_basic_types(context: ParseContext) -> Optional[str]:
    if context.origin in {int, float, str, bool}:
        return context.origin.__name__
    return None


@type_parser
def _parse_dict(context: ParseContext) -> Optional[str]:
    if issubclass(context.origin, Mapping):
        dict_args = get_args(context.annotation)
        if dict_args:
            key_type = dict_args[0]
            if key_type is not str:
                raise TypeError("Only \"str\" can be used as the type of dict keys.")
            key_type = "str"
            value_type = parse_type(dict_args[1], context.schemas)
            if isinstance(value_type, list):
                value_type = "|".join(value_type)
                # raise TypeError("\"Union\" cannot be used as the type of dict elements.")
            return f"Dict[{key_type},{value_type}]"
        else:
            return "Dict"
    return None


@type_parser
def _parse_list(context: ParseContext) -> Optional[str]:
    if issubclass(context.origin, Sequence):
        list_args = get_args(context.annotation)
        if list_args:
            if len(list_args) > 1:
                raise TypeError("Only ONE type can be used as the type of list elements.")
            elem_type = parse_type(list_args[0], context.schemas)
            if isinstance(elem_type, list):
                elem_type = "|".join(elem_type)
                # raise TypeError("\"Union\" cannot be used as the type of list elements.")
            return f"List[{elem_type}]"
        else:
            return "List"
    return None


@type_parser
def _parse_enum(context: ParseContext) -> Optional[str]:
    if issubclass(context.origin, enum.Enum):
        return f"Enum[{','.join(e.name for e in context.origin)}]"
    return None


@type_parser
def _parse_base_model(context: ParseContext) -> Optional[str]:
    origin = context.origin
    if issubclass(origin, BaseModel):
        _module = origin.__module__
        _name = origin.__name__
        model_name = _name if (_module is None) else f"{_module}.{_name}"

        is_new_model = model_name not in context.schemas
        is_not_base_class = origin is not BaseModel
        if is_new_model and is_not_base_class:
            schema = Schema(name=model_name)
            context.schemas[model_name] = schema
            # Once the Schema object is created, it should be put into the context immediately.
            # This is to prevent this object from being repeatedly parsed.
            # Repeated parsing will cause dead recursion!

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

        return model_name
    return None


@type_parser
def _parse_iterable(context: ParseContext) -> Optional[str]:
    if context.origin.__name__ in {"Iterable", "Generator", "range"} and issubclass(context.origin, Iterable):
        iter_args = get_args(context.annotation)
        if len(iter_args) < 1:
            raise TypeError("At least ONE type should be specified for iterable elements.")
        iter_type = parse_type(iter_args[0], context.schemas)
        if isinstance(iter_type, list):
            raise TypeError("\"Union\" cannot be used as the type of iterable elements.")
        return f"Iter[{iter_type}]"
    return None


@type_parser
def _parse_none_type(context: ParseContext) -> Optional[str]:
    origin = context.origin
    if origin.__module__ == "builtins" and origin.__name__ == "NoneType":
        return "NoneType"
    return None


@type_parser
def _parse_ndarray(context: ParseContext) -> Optional[str]:
    origin = context.origin
    if origin.__module__ == "numpy" and origin.__name__ == "ndarray":
        return "numpy.ndarray"
    return None


def generic_parser(fn):
    _GENERIC_PARSERS.append(fn)
    return fn


@generic_parser
def _parse_any(context: ParseContext) -> Optional[str]:
    if context.origin is Any or str(context.origin) == str(Any):
        return "Any"
    return None


@generic_parser
def _parse_union(context: ParseContext) -> Optional[List[str]]:
    if context.origin is Union or str(context.origin) == str(Union):
        return [
            parse_type(arg, context.schemas)
            for arg in get_args(context.annotation)
        ]
    return None


@generic_parser
def _parse_literal(context: ParseContext) -> Optional[str]:
    if context.origin is Literal or str(context.origin) == str(Literal):
        enum_args = get_args(context.annotation)
        return f"Enum[{','.join(map(str, enum_args))}]"
    return None


class QueryAPIOutput(BaseModel):
    input_schema: str
    output_schema: str
    context: Mapping[str, Schema]
    bundled_input: bool


def query_api(obj) -> QueryAPIOutput:
    api_models = obj if isinstance(obj, APISignature) else get_api_signature(obj)

    context = {}

    args_model = api_models.input_model or api_models.bundled_model
    input_schema = parse_type(args_model, context)

    output_schema = None
    if api_models.output_type is not None:
        output_schema = parse_type(api_models.output_type, context)
    if isinstance(output_schema, list):
        output_schema = output_schema[0]

    # output_schema = None
    # sig = signature(fn)
    # return_annotation = sig.return_annotation
    # if return_annotation is not None and return_annotation is not NoReturn:
    #     if return_annotation is sig.empty:
    #         return_annotation = Any
    #     output_schema = parse_type(return_annotation, context)
    # if isinstance(output_schema, list):
    #     output_schema = output_schema[0]

    return QueryAPIOutput(
        input_schema=input_schema,
        output_schema=output_schema,
        context=context,
        bundled_input=api_models.bundled_model is not None,
    )
