"""Ingredient metadata registry for tracking IDs, names, and matrix indices."""

import warnings

import numpy as np


class IngredientRegistry:
    """
    Central registry mapping matrix indices ↔ ingredient IDs ↔ names.

    This class provides a single source of truth for ingredient metadata,
    replacing the previous pattern of passing around separate id_to_index,
    index_to_id, and id_to_name dictionaries.

    Parameters
    ----------
    ingredients : list of tuple
        List of (matrix_index, ingredient_id, ingredient_name) tuples.
        Matrix indices must be contiguous integers starting at 0.

    Attributes
    ----------
    The class is immutable after construction. All lookups are O(1).

    Examples
    --------
    >>> ingredients = [(0, "123", "Gin"), (1, "456", "Vodka")]
    >>> registry = IngredientRegistry(ingredients)
    >>> registry.get_name(index=0)
    'Gin'
    >>> registry.get_index(id="456")
    1
    >>> len(registry)
    2
    """

    def __init__(self, ingredients: list[tuple[int, str, str]]):
        """
        Initialize the registry with ingredient data.

        Parameters
        ----------
        ingredients : list of tuple
            List of (matrix_index, ingredient_id, ingredient_name) tuples.

        Raises
        ------
        ValueError
            If matrix indices are not contiguous 0..N-1, or if duplicate IDs found.
        """
        self._validate_ingredients(ingredients)

        # Sort by index to ensure arrays are in matrix order
        ingredients_sorted = sorted(ingredients, key=lambda x: x[0])

        # Store as parallel numpy arrays for fast index-based access
        self._indices = np.array([idx for idx, _, _ in ingredients_sorted], dtype=int)
        self._ids = np.array([str(id) for _, id, _ in ingredients_sorted], dtype=object)
        self._names = np.array(
            [str(name) for _, _, name in ingredients_sorted], dtype=object
        )

        # Build reverse lookup dicts
        self._id_to_idx = {str(id): idx for idx, id, _ in ingredients_sorted}
        self._name_to_idx: dict[str, int] | None = (
            None  # Lazy initialization on first use
        )

    def _validate_ingredients(self, ingredients: list[tuple[int, str, str]]) -> None:
        """
        Validate ingredient data quality.

        Parameters
        ----------
        ingredients : list of tuple
            List of (matrix_index, ingredient_id, ingredient_name) tuples.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if not ingredients:
            raise ValueError("ingredients list cannot be empty")

        # Extract components
        indices = [idx for idx, _, _ in ingredients]
        ids = [str(id) for _, id, _ in ingredients]
        names = [str(name) for _, _, name in ingredients]

        # 1. Matrix indices must be contiguous 0..N-1
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(
                f"Matrix indices must be contiguous 0..{len(indices) - 1}. "
                f"Got: {sorted(indices)}"
            )

        # 2. IDs must be unique
        if len(ids) != len(set(ids)):
            duplicates = {id for id in ids if ids.count(id) > 1}
            raise ValueError(f"Duplicate ingredient IDs found: {duplicates}")

        # 3. Names should be unique (warn if not, don't fail)
        if len(names) != len(set(names)):
            duplicates = {name for name in names if names.count(name) > 1}
            warnings.warn(f"Duplicate ingredient names found: {duplicates}")

    def get_name(self, index: int | None = None, id: str | int | None = None) -> str:
        """
        Get ingredient name from either index or id.

        Parameters
        ----------
        index : int, optional
            Matrix index (0-based). Mutually exclusive with `id`.
        id : str, optional
            Ingredient ID. Mutually exclusive with `index`.

        Returns
        -------
        str
            Ingredient name.

        Raises
        ------
        ValueError
            If both or neither argument is provided.
        IndexError
            If index is out of range.
        KeyError
            If id is not found.

        Examples
        --------
        >>> registry.get_name(index=0)
        'Gin'
        >>> registry.get_name(id="123")
        'Gin'
        """
        if (index is None) == (id is None):
            raise ValueError("Exactly one of 'index' or 'id' must be provided")
        if isinstance(id, int):
            id = str(id)
        if index is not None:
            if not 0 <= index < len(self):
                raise IndexError(f"Index {index} out of range [0, {len(self)})")
            return str(self._names[index])
        else:
            if id not in self._id_to_idx:
                raise KeyError(f"Ingredient ID '{id}' not found in registry")
            return str(self._names[self._id_to_idx[id]])

    def get_id(self, *, index: int | None = None, name: str | None = None) -> str:
        """
        Get ingredient ID from either index or name.

        Parameters
        ----------
        index : int, optional
            Matrix index (0-based). Mutually exclusive with `name`.
        name : str, optional
            Ingredient name. Mutually exclusive with `index`.

        Returns
        -------
        str
            Ingredient ID.

        Raises
        ------
        ValueError
            If both or neither argument is provided.
        IndexError
            If index is out of range.
        KeyError
            If name is not found.

        Examples
        --------
        >>> registry.get_id(index=0)
        '123'
        >>> registry.get_id(name="Gin")
        '123'
        """
        if (index is None) == (name is None):
            raise ValueError("Exactly one of 'index' or 'name' must be provided")

        if index is not None:
            if not 0 <= index < len(self):
                raise IndexError(f"Index {index} out of range [0, {len(self)})")
            return str(self._ids[index])
        else:
            # Lazy build name index on first name lookup
            if self._name_to_idx is None:
                self._name_to_idx = {
                    str(name): idx for idx, name in enumerate(self._names)
                }
            if name not in self._name_to_idx:
                raise KeyError(f"Ingredient name '{name}' not found in registry")
            return str(self._ids[self._name_to_idx[name]])

    def get_index(self, *, id: str | None = None, name: str | None = None) -> int:
        """
        Get matrix index from either id or name.

        Parameters
        ----------
        id : str, optional
            Ingredient ID. Mutually exclusive with `name`.
        name : str, optional
            Ingredient name. Mutually exclusive with `id`.

        Returns
        -------
        int
            Matrix index (0-based).

        Raises
        ------
        ValueError
            If both or neither argument is provided.
        KeyError
            If id or name is not found.

        Examples
        --------
        >>> registry.get_index(id="123")
        0
        >>> registry.get_index(name="Gin")
        0
        """
        if (id is None) == (name is None):
            raise ValueError("Exactly one of 'id' or 'name' must be provided")

        if id is not None:
            if id not in self._id_to_idx:
                raise KeyError(f"Ingredient ID '{id}' not found in registry")
            return int(self._id_to_idx[id])
        else:
            # Lazy build name index on first name lookup
            if self._name_to_idx is None:
                self._name_to_idx = {
                    str(name): idx for idx, name in enumerate(self._names)
                }
            if name not in self._name_to_idx:
                raise KeyError(f"Ingredient name '{name}' not found in registry")
            return int(self._name_to_idx[name])

    def __len__(self) -> int:
        """Return the number of ingredients in the registry."""
        return len(self._ids)

    def __getitem__(self, index: int) -> tuple[str, str]:
        """
        Get (id, name) tuple for a matrix index.

        Parameters
        ----------
        index : int
            Matrix index (0-based).

        Returns
        -------
        tuple of str
            (ingredient_id, ingredient_name)

        Examples
        --------
        >>> registry[0]
        ('123', 'Gin')
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")
        return (str(self._ids[index]), str(self._names[index]))

    def validate_matrix(self, matrix: np.ndarray) -> None:
        """
        Validate that a matrix has compatible dimensions.

        Parameters
        ----------
        matrix : np.ndarray
            Square matrix to validate.

        Raises
        ------
        ValueError
            If matrix shape is incompatible with ingredient count.

        Examples
        --------
        >>> registry.validate_matrix(np.zeros((2, 2)))  # OK
        >>> registry.validate_matrix(np.zeros((3, 3)))  # Raises ValueError
        """
        if len(matrix.shape) != 2:
            raise ValueError(f"Matrix must be 2-dimensional, got shape {matrix.shape}")
        if matrix.shape[0] != len(self) or matrix.shape[1] != len(self):
            raise ValueError(
                f"Matrix shape {matrix.shape} incompatible with "
                f"{len(self)} ingredients (expected ({len(self)}, {len(self)}))"
            )

    def to_id_to_index(self) -> dict[str, int]:
        """
        Export as {ingredient_id: matrix_index} dict for legacy compatibility.

        Returns
        -------
        dict of str to int
            Copy of the ID to index mapping.
        """
        return self._id_to_idx.copy()
