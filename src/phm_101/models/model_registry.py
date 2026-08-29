from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from phm_101.data_types.enums import ModelName

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from phm_101.config.configs import DataConfig, ModelConfig

T = TypeVar('T', bound=type)


class ModelRegistry:
    """Registry for model architectures.

    Each architecture is registered under the name its config carries.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: ModelName) -> Callable[[T], T]:
        """Declare a model under the name its config carries."""

        def decorator(architecture: T) -> T:
            key = ModelName(name).value
            cls._registry[key] = architecture
            return architecture

        return decorator

    @classmethod
    def get_model(
        cls, model_config: ModelConfig, data_config: DataConfig
    ) -> nn.Module:
        """Instantiate the architecture registered under the config's name."""
        # importing the package runs the decorators that fill the registry
        import phm_101.models  # noqa: F401, PLC0415

        architecture = cls._registry.get(ModelName(model_config.name).value)
        if architecture is None:
            raise ValueError(
                f'unknown model {model_config.name!r}; '
                f'registered: {sorted(cls._registry)}'
            )
        return architecture(model_config, data_config)
