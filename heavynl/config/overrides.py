from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, ClassVar, Optional

from confz.change import SourceChangeManager
from confz.confz_source import ConfZSources, FileFormat
from confz.exceptions import ConfZException, ConfZFileException
from confz.loaders.file_loader import FileLoader
from pydantic import BaseModel

# These overrides are responsible for modifying the default behaviour of the ConfZ library.
# 1. Treating a .conf file as TOML
# 2. Allowing the ConfZ class to be mutable


class OverrideFileLoader(FileLoader):
    @classmethod
    def _get_format(cls, file_path: Path, file_format: Optional[FileFormat]) -> FileFormat:
        if file_format is not None:
            return file_format

        suffix_formats = {
            ".yml": FileFormat.YAML,
            ".yaml": FileFormat.YAML,
            ".json": FileFormat.JSON,
            ".toml": FileFormat.TOML,
            ".conf": FileFormat.TOML,  # 1. Treat .conf as TOML
        }
        suffix = file_path.suffix
        try:
            suffix_format = suffix_formats[suffix]
        except KeyError as e:
            raise ConfZFileException(
                f"File-ending '{suffix}' is not known. Supported are: " f"{', '.join(list(suffix_formats.keys()))}."
            ) from e

        return suffix_format


def _load_config(config_kwargs: dict, confz_sources: ConfZSources) -> dict:
    config = config_kwargs.copy()
    if isinstance(confz_sources, list):
        for confz_source in confz_sources:
            loader = OverrideFileLoader
            loader.populate_config(config, confz_source)  # type: ignore
    else:
        loader = OverrideFileLoader
        loader.populate_config(config, confz_sources)  # type: ignore
    return config


# Metaclass of pydantic.BaseModel is not in __all__, so use type(BaseModel).
# ConfZ will be only class with this meta class.
# Both of these things confuse mypy and pylint, so had to disable multiple times.
class OverrideConfZMetaclass(type(BaseModel)):  # type: ignore
    """ConfZ Meta Class, inheriting from the pydantic `BaseModel` MetaClass."""

    # pylint: disable=no-self-argument,no-member
    def __call__(cls, config_sources: Optional[ConfZSources] = None, **kwargs):
        """Called every time an instance of any ConfZ object is created. Injects the
        config value population and singleton mechanism."""
        if config_sources is not None:
            config = _load_config(kwargs, config_sources)
            return super().__call__(**config)

        if cls.CONFIG_SOURCES is not None:  # type: ignore
            # pylint: disable=access-member-before-definition
            # pylint: disable=attribute-defined-outside-init
            if len(kwargs) > 0:
                raise ConfZException(
                    'Singleton mechanism enabled ("CONFIG_SOURCES" is defined), so '
                    "keyword arguments are not supported"
                )
            if cls.confz_instance is None:  # type: ignore
                config = _load_config(kwargs, cls.CONFIG_SOURCES)  # type: ignore
                cls.confz_instance = super().__call__(**config)
            return cls.confz_instance

        return super().__call__(**kwargs)


class OverrideConfZ(BaseModel, metaclass=OverrideConfZMetaclass):
    """Base class, parent of every config class. Internally wraps :class:`BaseModel`of
    pydantic and behaves transparent except for two cases:

    - If the constructor gets `config_sources` as kwarg, these sources are used as
      input to enrich the other kwargs.
    - If the class has the class variable `CONFIG_SOURCES` defined, these sources are
      used as input.

    In the latter case, a singleton mechanism is activated, returning the same config
    class instance every time the constructor is called."""

    CONFIG_SOURCES: ClassVar[Optional[ConfZSources]] = None  #: Sources to use as input.

    # type is ClassVar[Optional["ConfZ"]] (pydantic throws error with forward ref)
    confz_instance: ClassVar[Optional[Any]] = None  #: *for internal use only*

    # type is ClassVar[Optional[List["Listener"]]] (same here)
    listeners: ClassVar[Optional[list[Any]]] = None  #: *for internal use only*

    class Config:
        allow_mutation = True  # 2. Allow ConfZ to be mutable

    @classmethod
    def change_config_sources(cls, config_sources: ConfZSources) -> AbstractContextManager:
        """Change the `CONFIG_SOURCES` class variable within a controlled context.
        Within this context, the sources will be different and the singleton reset.
        This can be useful in unit tests to temporarily change a configuration.

        :param config_sources: The temporary config sources for within the context.
        :return: Context manager for change of config sources.
        """
        return SourceChangeManager(cls, config_sources)
