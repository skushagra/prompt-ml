from __future__ import annotations

from typing import Any, Iterator


class Context:
    """
    Shared, mutable data store that flows through every instruction in a
    workflow.

    Data is first-class in prompt-ml: every instruction — whether it
    collects data, classifies input, generates output, or fires a side
    effect — reads from and writes to the same ``Context``.  This means
    a ``ClassifyInstruction`` can inspect what a prior
    ``DataCollectionInstruction`` collected, an ``ActionInstruction`` can
    read every field collected so far, and an ``OutputInstruction`` can
    interpolate any value into its template.

    Structure
    ---------
    Keys are plain strings.  By convention, use dotted namespacing to
    avoid collisions across instructions::

        context["address.line1"]     # written by CollectAddress
        context["intent"]            # written by ClassifyIntent
        context["booking.result"]    # written by BookAppointment action

    A ``ContextView`` (returned by :meth:`namespace`) provides a scoped
    wrapper that prepends the namespace automatically, so instruction code
    can use short keys without worrying about collisions.

    Usage
    -----
    ::

        ctx = Context()
        ctx["address.line1"] = "742 Evergreen Terrace"
        ctx["intent"] = "service"

        # Scoped view
        addr = ctx.namespace("address")
        addr["city"] = "Springfield"           # sets "address.city"
        print(addr["city"])                    # reads "address.city"

        # Snapshot for template interpolation
        ctx.snapshot()                         # {"address.line1": ..., ...}

        # Flat snapshot — dotted keys flattened to nested dicts
        ctx.as_nested()                        # {"address": {"line1": ...}, ...}
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def namespace(self, prefix: str) -> "ContextView":
        """
        Return a scoped view whose keys are automatically prefixed with
        ``prefix + "."``::

            view = ctx.namespace("address")
            view["city"] = "Springfield"   # stores "address.city"
        """
        return ContextView(self, prefix)

    def update(self, data: dict[str, Any]) -> None:
        """Merge ``data`` into the context."""
        self._data.update(data)

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of all key-value pairs."""
        return dict(self._data)

    def as_nested(self) -> dict[str, Any]:
        """
        Return a nested dict by expanding dotted keys::

            {"address.city": "Springfield", "intent": "service"}
            →  {"address": {"city": "Springfield"}, "intent": "service"}
        """
        nested: dict[str, Any] = {}
        for key, value in self._data.items():
            parts = key.split(".")
            node = nested
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return nested

    def __repr__(self) -> str:
        return f"Context({self._data!r})"


class ContextView:
    """
    A scoped, prefix-prepending view over a parent :class:`Context`.

    All reads and writes transparently add ``prefix + "."`` to every key,
    keeping instruction code free of namespace boilerplate.
    """

    def __init__(self, parent: Context, prefix: str) -> None:
        self._parent = parent
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}.{key}"

    def get(self, key: str, default: Any = None) -> Any:
        return self._parent.get(self._key(key), default)

    def set(self, key: str, value: Any) -> None:
        self._parent.set(self._key(key), value)

    def __getitem__(self, key: str) -> Any:
        return self._parent[self._key(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._parent[self._key(key)] = value

    def __contains__(self, key: str) -> bool:
        return self._key(key) in self._parent

    def snapshot(self) -> dict[str, Any]:
        """All key-value pairs under this namespace (keys without prefix)."""
        prefix = self._prefix + "."
        return {
            k[len(prefix):]: v
            for k, v in self._parent.snapshot().items()
            if k.startswith(prefix)
        }

    def __repr__(self) -> str:
        return f"ContextView(prefix={self._prefix!r}, data={self.snapshot()!r})"
