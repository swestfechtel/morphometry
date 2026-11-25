"""Static import closure of a package's modules, computed with ``ast`` (standard library only).

The release tooling needs to know which files an image actually *runs*: the torsion image
ships the whole ``morphometry/`` folder, but only the modules reachable from its entry point
can change its numbers. :func:`import_closure` walks ``import`` statements (at any nesting
depth, so imports inside functions count) from a set of entry files and returns every file of
one package it reaches.

The walker is deliberately conservative and loud: a ``package`` import it cannot map to a
file, and any ``importlib.import_module`` / ``__import__`` call whose literal argument names
the package, raise :class:`UnresolvedImport` instead of silently shrinking the closure.

Not seen (so not guarded): an aliased ``import_module`` (``im = importlib.import_module``),
``runpy``, ``importlib.util.spec_from_file_location`` and non-literal module names.
"""
import ast
from pathlib import Path

# Call names whose first argument is a dotted module name imported at runtime.
_DYNAMIC_IMPORT_CALLS = frozenset({"import_module", "__import__"})


class UnresolvedImport(Exception):
    """A ``package`` import the static walker cannot map to a file.

    :param path: absolute path of the file holding the import.
    :param lineno: 1-based line number of the import statement or call.
    :param name: the dotted name that could not be resolved (for a dynamic import, the
        literal text of its argument).
    """

    def __init__(self, path: Path, lineno: int, name: str):
        self.path = path
        self.lineno = lineno
        self.name = name
        super().__init__(f"{path}:{lineno}: cannot resolve import of {name!r}")


def _is_in_package(dotted: str, package: str) -> bool:
    """Tell whether a dotted module name lies inside ``package``.

    :param dotted: absolute dotted module name, e.g. ``"morphometry.femur"``.
    :param package: top-level package name, e.g. ``"morphometry"``.
    :return: ``True`` for ``package`` itself and any of its submodules.
    """
    return dotted == package or dotted.startswith(package + ".")


def _bound_names(tree: ast.Module) -> tuple[set[str], bool]:
    """Collect the names a module binds at module scope.

    Statements nested in ``if`` / ``try`` / ``with`` / loops are included (they run at import
    time); function and class bodies are not.

    :param tree: the parsed module.
    :return: ``(names, has_star)`` — the bound names, and whether the module performs a
        ``from ... import *`` (in which case any name may be bound and none can be refuted).
    """
    names: set[str] = set()
    has_star = False

    def add_target(target: ast.AST) -> None:
        # Assignment targets may be tuples/lists/starred; attributes and subscripts bind nothing.
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)
        elif isinstance(target, ast.Starred):
            add_target(target.value)

    def visit(body: list[ast.stmt]) -> None:
        nonlocal has_star
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    add_target(target)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                add_target(node.target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        has_star = True
                    else:
                        names.add(alias.asname or alias.name)
            # Recurse into compound statements that execute at import time.
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list) and not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    visit(inner)
            for handler in getattr(node, "handlers", []) or []:
                if handler.name:
                    names.add(handler.name)
                visit(handler.body)
            if isinstance(node, (ast.For, ast.AsyncFor)):
                add_target(node.target)
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)

    visit(tree.body)
    return names, has_star


class _Walker:
    """Worklist state for one :func:`import_closure` call; each file is parsed at most once.

    :param repo: repository root that dotted names are resolved against.
    :param package: top-level package whose modules are collected.
    """

    def __init__(self, repo: Path, package: str):
        self.repo = repo
        self.package = package
        self.closure: set[Path] = set()  # repo-relative files in the closure
        self.queued: set[Path] = set()  # absolute files already scheduled for parsing
        self.worklist: list[Path] = []
        self.trees: dict[Path, ast.Module] = {}
        self.names: dict[Path, tuple[set[str], bool]] = {}

    def parse(self, path: Path) -> ast.Module:
        """Parse a file once and cache its tree.

        :param path: absolute path of a Python file.
        :return: the parsed module.
        """
        if path not in self.trees:
            self.trees[path] = ast.parse(path.read_bytes(), filename=str(path))
        return self.trees[path]

    def module_location(self, dotted: str) -> Path | None | bool:
        """Map a dotted module name to its file.

        :param dotted: absolute dotted module name.
        :return: the absolute path of ``<parts>.py`` or ``<parts>/__init__.py``; ``True`` for a
            namespace package (a directory without ``__init__.py``); ``None`` if nothing exists.
        """
        base = self.repo.joinpath(*dotted.split("."))
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
        if base.is_dir():
            return True
        return None

    def add_file(self, path: Path) -> None:
        """Record a file in the closure and schedule it for parsing.

        :param path: absolute path of a package file.
        """
        self.closure.add(path.relative_to(self.repo))
        if path not in self.queued:
            self.queued.add(path)
            self.worklist.append(path)

    def add_module(self, dotted: str, source: Path, lineno: int) -> None:
        """Add a module and the ``__init__.py`` of every package it is imported through.

        :param dotted: absolute dotted module name inside the package.
        :param source: file holding the import (for error reporting).
        :param lineno: line of the import (for error reporting).
        :raises UnresolvedImport: if the module or one of its parent packages does not exist.
        """
        parts = dotted.split(".")
        for depth in range(1, len(parts) + 1):
            location = self.module_location(".".join(parts[:depth]))
            if location is None:
                raise UnresolvedImport(source, lineno, dotted)
            if isinstance(location, Path):
                self.add_file(location)

    def has_name(self, dotted: str, name: str) -> bool:
        """Tell whether an existing module binds ``name`` at module scope.

        :param dotted: absolute dotted module name (must already resolve).
        :param name: the imported name.
        :return: ``True`` if bound, or if the module star-imports (cannot be refuted).
        """
        location = self.module_location(dotted)
        if not isinstance(location, Path):
            return False  # a namespace package binds no names of its own
        if location not in self.names:
            self.names[location] = _bound_names(self.parse(location))
        names, has_star = self.names[location]
        return has_star or name in names

    def current_package(self, path: Path) -> list[str]:
        """Dotted parts of the package a file's relative imports are resolved against.

        :param path: absolute path of the importing file.
        :return: the parts of its directory relative to the repo root (for ``pkg/__init__.py``
            and ``pkg/mod.py`` alike this is ``["pkg"]``).
        """
        return list(path.relative_to(self.repo).parent.parts)

    def visit_file(self, path: Path) -> None:
        """Resolve every import statement and dynamic-import call in one file.

        :param path: absolute path of the file to scan.
        :raises UnresolvedImport: see :func:`import_closure`.
        """
        for node in ast.walk(self.parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_in_package(alias.name, self.package):
                        self.add_module(alias.name, path, node.lineno)
            elif isinstance(node, ast.ImportFrom):
                self.visit_import_from(node, path)
            elif isinstance(node, ast.Call):
                self.check_dynamic(node, path)

    def visit_import_from(self, node: ast.ImportFrom, path: Path) -> None:
        """Resolve ``from X import Y``: ``X.Y`` if it is a module, else ``Y`` must be a name in ``X``.

        :param node: the ``ImportFrom`` node.
        :param path: absolute path of the importing file.
        :raises UnresolvedImport: if neither reading resolves, and for any ``from X import *``.
        """
        if node.level:
            base = self.current_package(path)
            if node.level - 1 > len(base):
                raise UnresolvedImport(path, node.lineno, "." * node.level + (node.module or ""))
            base = base[: len(base) - (node.level - 1)]
            target = ".".join(base + ([node.module] if node.module else []))
        else:
            target = node.module or ""
        if not _is_in_package(target, self.package):
            return
        for alias in node.names:
            if alias.name == "*":
                # Conservative: which names (and submodules) a star import binds is not checked.
                raise UnresolvedImport(path, node.lineno, f"{target}.*")
            submodule = f"{target}.{alias.name}"
            if self.module_location(submodule) is not None:
                self.add_module(submodule, path, node.lineno)
                continue
            self.add_module(target, path, node.lineno)
            if not self.has_name(target, alias.name):
                raise UnresolvedImport(path, node.lineno, f"{target}.{alias.name}")

    def check_dynamic(self, node: ast.Call, path: Path) -> None:
        """Reject ``import_module`` / ``__import__`` calls whose literal name or package names the package.

        Non-literal arguments are not inspected: they cannot be resolved statically, and the
        codebase's only such call (``api/tasks/queue.py``) resolves ``api`` job functions.

        :param node: a call node.
        :param path: absolute path of the file holding the call.
        :raises UnresolvedImport: for a dynamic import of the package.
        """
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called not in _DYNAMIC_IMPORT_CALLS:
            return
        # The module name (first positional or ``name=``) and importlib's ``package`` anchor
        # (second positional or ``package=``), e.g. ``import_module(".femur", "morphometry")``.
        candidates = list(node.args[:2]) + [
            kw.value for kw in node.keywords if kw.arg in ("name", "package")
        ]
        for arg in candidates:
            text = None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                text = arg.value
            elif isinstance(arg, ast.JoinedStr) and arg.values:
                head = arg.values[0]
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    text = head.value
            if text is not None and (text == self.package or text.startswith(self.package + ".")):
                raise UnresolvedImport(path, node.lineno, text)


def import_closure(entry_files: list[Path], repo: Path, package: str = "morphometry") -> set[Path]:
    """Compute every file of ``package`` reachable from ``entry_files`` through imports.

    Followed: ``import x.y``, ``from x import y`` (``y`` a submodule or a name defined in ``x``),
    relative imports, imports at any nesting depth (inside functions, ``try`` blocks, …), and
    the ``__init__.py`` of every package traversed. Imports of other packages are ignored.

    :param entry_files: files to start from (absolute, or relative to ``repo``); they must lie
        inside ``repo``. An entry file is itself part of the result only if it lies inside
        ``package``'s directory.
    :param repo: repository (or exported source tree) root that dotted names resolve against.
    :param package: name of the top-level package to collect, default ``"morphometry"``.
    :return: repo-relative paths of the closure's files.
    :raises UnresolvedImport: for a ``package`` import that maps to no file (or to no name in an
        existing module), for ``from <package...> import *``, and for an
        ``importlib.import_module`` / ``__import__`` call whose literal name or ``package``
        argument starts with ``package``.
    """
    repo = Path(repo).resolve()
    walker = _Walker(repo, package)
    for entry in entry_files:
        entry = (repo / entry).resolve()
        if entry.relative_to(repo).parts[:1] == (package,):
            walker.add_file(entry)
        elif entry not in walker.queued:
            walker.queued.add(entry)
            walker.worklist.append(entry)
    while walker.worklist:
        walker.visit_file(walker.worklist.pop())
    return walker.closure
