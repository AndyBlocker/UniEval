"""Universal Registry class for all extension points."""


class Registry:
    """A generic registry for registering and retrieving components by key.

    Used for quantizers, neurons, conversion_rules, evaluators, ops_hooks,
    and model_profiles.
    """

    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, key):
        """Decorator to register a component under the given key."""
        def decorator(obj):
            if key in self._registry:
                raise KeyError(
                    f"Key '{key}' already registered in {self.name} registry"
                )
            self._registry[key] = obj
            return obj
        return decorator

    def register_obj(self, key, obj):
        """Directly register an object (non-decorator form)."""
        if key in self._registry:
            raise KeyError(
                f"Key '{key}' already registered in {self.name} registry"
            )
        self._registry[key] = obj

    def get(self, key):
        """Retrieve a registered component by key."""
        if key not in self._registry:
            raise KeyError(
                f"Key '{key}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def list_keys(self):
        """Return all registered keys."""
        return list(self._registry.keys())

    def __contains__(self, key):
        return key in self._registry

    def __repr__(self):
        return f"Registry(name={self.name}, keys={self.list_keys()})"


# Global registries
QUANTIZER_REGISTRY = Registry("quantizers")
NEURON_REGISTRY = Registry("neurons")
EVALUATOR_REGISTRY = Registry("evaluators")
MODEL_PROFILE_REGISTRY = Registry("model_profiles")
