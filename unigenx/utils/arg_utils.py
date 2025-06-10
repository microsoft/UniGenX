# -*- coding: utf-8 -*-
import ast
import typing
from argparse import ArgumentParser
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from enum import Enum
from typing import List, Type

import yaml

from unigenx.logging import logger


def make_enum_praser(enum: Enum):
    choices = [e.name for e in enum]

    def parse_enum(arg):
        try:
            return enum[arg]
        except KeyError:
            raise ValueError(
                f"Invalid choice: {arg} for type {enum.__name__}. Valid choices are: {choices}"
            )

    return parse_enum


def argument_exists(parser, arg_name):
    if arg_name.startswith("--"):
        arg_name = arg_name[2:]

    # args_name_2 = arg_name.replace("_", "-")
    args_name_pre = "--" + arg_name
    # args_name_2_pre = "--" + args_name_2

    return (
        arg_name in parser._option_string_actions
        # or args_name_2 in parser._option_string_actions
        or args_name_pre in parser._option_string_actions
        # or args_name_2_pre in parser._option_string_actions
    )


def unwarp_optional(field_type: typing.Type):
    if typing.get_origin(field_type) == typing.Union:
        args = typing.get_args(field_type)
        if len(args) == 2 and args[1] == type(None) and args[0] != type(None):
            return args[0]
    return field_type


def is_enum_type(tp: typing.Type) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)


def is_collection(tp: typing.Type) -> bool:
    if hasattr(tp, "__origin__"):
        tp = tp.__origin__

    return tp in (list, tuple, set)


def add_dataclass_to_parser(configs, parser: ArgumentParser):
    for config in configs:
        group = parser.add_argument_group(config.__name__)
        for field in fields(config):
            name = field.name.replace("-", "_")

            field_type = unwarp_optional(field.type)

            # if name in exist_configs:
            if argument_exists(parser, name):
                logger.warning(f"Duplicate config name: {name}, not added to parser")
                continue

            if field.default != MISSING:
                default = field.default
            elif field.default_factory != MISSING:
                default = field.default_factory()
            else:
                default = None

            if field_type == bool:
                group.add_argument("--" + name, action="store_true", default=default)
                group.add_argument("--no_" + name, action="store_false", dest=name)
            elif is_enum_type(field_type):
                parse_enum = make_enum_praser(field.type)
                group.add_argument("--" + name, type=parse_enum, default=default)
            elif is_collection(field_type):
                group.add_argument("--" + name, type=ast.literal_eval, default=default)
            else:
                group.add_argument("--" + name, type=field_type, default=default)

    return parser


def add_dataclass_to_dictconfig(configs: List[Type[dataclass]], config_path: str):
    fields = {
        field.name: (field.type, Field())
        for config in configs
        for field in fields(config)
    }
    Config = type("Config", (object,), fields)
    Config = dataclass(Config)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    args = Config(**data)

    return args


def from_args(args, config):
    kwargs = {}
    for field in fields(config):
        name = field.name.replace("-", "_")
        if isinstance(args, dict):
            value = args.get(name, None)
        else:
            value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    return config(**kwargs)


class ExtraArgsProvider:
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, parser: ArgumentParser):
        parsar = add_dataclass_to_parser(self.configs, parser)
        return parsar
