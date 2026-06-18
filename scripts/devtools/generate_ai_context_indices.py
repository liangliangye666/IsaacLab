#!/usr/bin/env python3
"""Generate AI-readable index files for the IsaacLab repository.

The script only reads repository files and writes generated artifacts under
docs/ai_context. It intentionally avoids importing IsaacLab modules so it can
run without launching Isaac Sim.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "ai_context"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".agents",
    ".codex",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "_build",
    "_isaac_sim",
    "build",
    "dist",
    "logs",
    "outputs",
    "venv",
}

EXCLUDED_DIR_SUFFIXES = {
    ".egg-info",
}

COLLAPSED_DIRS = {
    Path("docs/licenses"),
    Path("docs/source/_static"),
    Path("docs/source/refs/bibliography"),
    Path("source/isaaclab_assets/data"),
}

PY_SOURCE_ROOTS = [
    ROOT / "source" / "isaaclab" / "isaaclab",
    ROOT / "source" / "isaaclab_assets" / "isaaclab_assets",
    ROOT / "source" / "isaaclab_contrib" / "isaaclab_contrib",
    ROOT / "source" / "isaaclab_mimic" / "isaaclab_mimic",
    ROOT / "source" / "isaaclab_rl" / "isaaclab_rl",
    ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks",
]

TASK_REGISTRY_ROOTS = [
    ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks",
    ROOT / "source" / "isaaclab_mimic" / "isaaclab_mimic",
    ROOT / "source" / "isaaclab_contrib" / "isaaclab_contrib",
]

SYMBOL_SCRIPT_ROOTS = [
    ROOT / "scripts" / "reinforcement_learning",
    ROOT / "scripts" / "environments",
    ROOT / "scripts" / "sim2sim_transfer",
]

KEY_CLASS_RE = re.compile(
    r"(Env|EnvCfg|Cfg|Manager|Action|Command|Observation|Reward|Termination|Curriculum|Scene|Asset|Articulation|"
    r"RigidObject|Sensor|Camera|Controller|Wrapper|Runner|Term|Terrain|Importer|Converter|Spawner|Actuator|Robot|"
    r"Policy|Algorithm|Mimic|Direct|RslRl|PPO)"
)

DIRECT_ENV_HOOKS = {
    "_setup_scene",
    "_pre_physics_step",
    "_apply_action",
    "_get_observations",
    "_get_rewards",
    "_get_dones",
    "_reset_idx",
    "_set_debug_vis_impl",
    "_debug_vis_callback",
    "compute_rewards",
}

UTILITY_FUNCTIONS = {
    "load_cfg_from_registry",
    "parse_env_cfg",
    "get_checkpoint_path",
    "hydra_task_config",
    "multi_agent_to_single_agent",
    "export_policy_as_jit",
    "export_policy_as_onnx",
}


@dataclass
class RegistryEntry:
    task_id: str
    file: Path
    line: int
    entry_point: str
    kwargs: dict[str, str]


@dataclass
class SymbolEntry:
    group: str
    file: Path
    line: int
    kind: str
    name: str
    detail: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def should_skip_dir(path: Path) -> bool:
    name = path.name
    if name in EXCLUDED_DIR_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in EXCLUDED_DIR_SUFFIXES)


def is_collapsed(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return False
    return any(relative == item for item in COLLAPSED_DIRS)


def iter_python_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(should_skip_dir(parent) for parent in path.parents):
                continue
            if "/test/" in f"/{rel(path)}/" or "/docs/" in f"/{rel(path)}/":
                continue
            files.append(path)
    return sorted(set(files))


def module_name_from_path(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if len(parts) >= 3 and parts[0] == "source":
        module_parts = list(parts[2:])
    elif parts[0] == "scripts":
        module_parts = list(parts)
    else:
        module_parts = list(parts)
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts)


def package_name_for_path(path: Path, module_name: str) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0] if "." in module_name else ""


def resolve_relative_module(current_package: str, level: int, module: str | None) -> str:
    parts = current_package.split(".") if current_package else []
    if level > 1:
        parts = parts[: -(level - 1)]
    base = ".".join(parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def build_import_map(tree: ast.AST, path: Path, module_name: str) -> dict[str, str]:
    imports = {"__name__": module_name}
    current_package = package_name_for_path(path, module_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative_module(current_package, node.level, node.module)
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "*":
                    continue
                if node.module is None:
                    imports[local_name] = f"{base}.{alias.name}" if base else alias.name
                else:
                    imports[local_name] = f"{base}.{alias.name}" if base else alias.name
    return imports


def eval_expr(node: ast.AST | None, module_name: str, imports: dict[str, str]) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Name):
        if node.id == "__name__":
            return module_name
        return imports.get(node.id, node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant):
                parts.append(str(item.value))
            elif isinstance(item, ast.FormattedValue):
                parts.append(eval_expr(item.value, module_name, imports))
        return "".join(parts)
    if isinstance(node, ast.Attribute):
        base = eval_expr(node.value, module_name, imports)
        if node.attr == "__name__":
            return base
        if ":" in base:
            return f"{base}.{node.attr}"
        if node.attr[:1].isupper() or node.attr.endswith("Cfg"):
            return f"{base}:{node.attr}"
        return f"{base}.{node.attr}"
    if isinstance(node, ast.Call):
        return ast.unparse(node)
    if isinstance(node, ast.Subscript):
        return ast.unparse(node)
    if isinstance(node, ast.Tuple | ast.List):
        return ", ".join(eval_expr(item, module_name, imports) for item in node.elts)
    return ast.unparse(node)


def eval_dict(node: ast.AST | None, module_name: str, imports: dict[str, str]) -> dict[str, str]:
    if not isinstance(node, ast.Dict):
        return {}
    result: dict[str, str] = {}
    for key_node, value_node in zip(node.keys, node.values):
        key = eval_expr(key_node, module_name, imports)
        result[key] = eval_expr(value_node, module_name, imports)
    return result


def is_gym_register_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "register"


def extract_registry_entries() -> list[RegistryEntry]:
    entries: list[RegistryEntry] = []
    for path in iter_python_files(TASK_REGISTRY_ROOTS):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module_name = module_name_from_path(path)
        imports = build_import_map(tree, path, module_name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not is_gym_register_call(node):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
            task_id = eval_expr(keywords.get("id") or (node.args[0] if node.args else None), module_name, imports)
            entry_point = eval_expr(keywords.get("entry_point"), module_name, imports)
            kwargs = eval_dict(keywords.get("kwargs"), module_name, imports)
            if task_id and task_id != "None":
                entries.append(RegistryEntry(task_id, path, node.lineno, entry_point, kwargs))
    return sorted(entries, key=lambda item: (item.task_id, rel(item.file), item.line))


def workflow_for_entry(entry: RegistryEntry) -> str:
    path = rel(entry.file)
    if "isaaclab_mimic" in path:
        return "mimic"
    if "/direct/" in path:
        return "direct"
    if "/manager_based/" in path:
        return "manager-based"
    if "isaaclab_contrib" in path:
        return "contrib"
    return "unknown"


def md(value: str) -> str:
    value = value.replace("\n", " ").strip()
    if not value or value == "None":
        return "-"
    return f"`{value}`"


def write_task_registry_index() -> None:
    entries = extract_registry_entries()
    lines = [
        "# IsaacLab Gym Task Registry Index",
        "",
        "Generated by `scripts/devtools/generate_ai_context_indices.py`.",
        "This file indexes static `gym.register(...)` calls without importing IsaacLab or launching Isaac Sim.",
        "",
        f"Total registered task specs found: **{len(entries)}**",
        "",
        "| Task id | Workflow | Env entry | RSL-RL | RL-Games | SKRL | SB3 | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        kwargs = entry.kwargs
        lines.append(
            "| "
            + " | ".join(
                [
                    md(entry.task_id),
                    workflow_for_entry(entry),
                    md(kwargs.get("env_cfg_entry_point", "")),
                    md(kwargs.get("rsl_rl_cfg_entry_point", "")),
                    md(kwargs.get("rl_games_cfg_entry_point", "")),
                    md(kwargs.get("skrl_cfg_entry_point", "")),
                    md(kwargs.get("sb3_cfg_entry_point", "")),
                    f"`{rel(entry.file)}:{entry.line}`",
                ]
            )
            + " |"
        )
    lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "04_TASK_REGISTRY_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def tree_entries(path: Path, prefix: str, depth: int, max_depth: int, lines: list[str]) -> None:
    if depth >= max_depth:
        return
    children = [child for child in path.iterdir() if not should_skip_dir(child)]
    children.sort(key=lambda p: (p.is_file(), p.name.lower()))
    for index, child in enumerate(children):
        connector = "`-- " if index == len(children) - 1 else "|-- "
        next_prefix = "    " if index == len(children) - 1 else "|   "
        suffix = "/" if child.is_dir() else ""
        if child.is_dir() and is_collapsed(child):
            lines.append(f"{prefix}{connector}{child.name}/ [collapsed]")
            continue
        lines.append(f"{prefix}{connector}{child.name}{suffix}")
        if child.is_dir():
            tree_entries(child, prefix + next_prefix, depth + 1, max_depth, lines)


def write_directory_tree(max_depth: int = 8) -> None:
    lines = [
        "IsaacLab directory tree",
        "Generated by scripts/devtools/generate_ai_context_indices.py",
        "",
        "Notes:",
        "- Excludes .git, caches, logs, outputs, build artifacts, virtualenvs, and egg-info directories.",
        "- Collapses large asset/license/static directories to keep this file AI-readable.",
        f"- Max recursion depth: {max_depth}.",
        "",
        "./",
    ]
    tree_entries(ROOT, "", 0, max_depth, lines)
    lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "02_DIRECTORY_TREE.txt").write_text("\n".join(lines), encoding="utf-8")


def base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for base in node.bases:
        try:
            names.append(ast.unparse(base))
        except Exception:
            names.append("<unknown>")
    return names


def class_group(path: Path) -> str:
    r = rel(path)
    if r.startswith("source/isaaclab/isaaclab/envs") or r.startswith("source/isaaclab/isaaclab/managers"):
        return "core_env_and_managers"
    if r.startswith("source/isaaclab/isaaclab/scene"):
        return "scene"
    if r.startswith("source/isaaclab/isaaclab/assets") or r.startswith("source/isaaclab_assets"):
        return "assets"
    if r.startswith("source/isaaclab/isaaclab/sim"):
        return "simulation"
    if r.startswith("source/isaaclab/isaaclab/sensors"):
        return "sensors"
    if r.startswith("source/isaaclab/isaaclab/controllers"):
        return "controllers"
    if r.startswith("source/isaaclab_rl"):
        return "rl_wrappers_and_cfg"
    if r.startswith("source/isaaclab_mimic"):
        return "mimic"
    if r.startswith("source/isaaclab_contrib"):
        return "contrib"
    if r.startswith("source/isaaclab_tasks"):
        return "tasks"
    if r.startswith("scripts/"):
        return "scripts"
    return "other"


def is_mdp_function(path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    r = rel(path)
    name = node.name
    return "/mdp/" in f"/{r}/" or name in DIRECT_ENV_HOOKS or name in UTILITY_FUNCTIONS


def extract_symbols() -> list[SymbolEntry]:
    entries: list[SymbolEntry] = []
    files = iter_python_files(PY_SOURCE_ROOTS) + iter_python_files(SYMBOL_SCRIPT_ROOTS)
    for path in sorted(set(files)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        group = class_group(path)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = base_names(node)
                signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
                if KEY_CLASS_RE.search(node.name) or any(KEY_CLASS_RE.search(base) for base in bases):
                    entries.append(SymbolEntry(group, path, node.lineno, "class", node.name, signature))
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, ast.FunctionDef) and item.name in DIRECT_ENV_HOOKS:
                        entries.append(
                            SymbolEntry(group, path, item.lineno, "method", f"{node.name}.{item.name}", ast.unparse(item.args))
                        )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if is_mdp_function(path, node):
                    entries.append(SymbolEntry(group, path, node.lineno, "function", node.name, ast.unparse(node.args)))
    return sorted(entries, key=lambda item: (item.group, rel(item.file), item.line, item.name))


def write_symbol_index() -> None:
    entries = extract_symbols()
    lines = [
        "IsaacLab raw symbol index",
        "Generated by scripts/devtools/generate_ai_context_indices.py",
        "",
        "Format:",
        "  group | file:line | kind | name | detail",
        "",
        f"Total symbols found: {len(entries)}",
        "",
    ]
    current_group = None
    for entry in entries:
        if entry.group != current_group:
            current_group = entry.group
            lines.append(f"[{current_group}]")
        lines.append(f"{rel(entry.file)}:{entry.line} | {entry.kind} | {entry.name} | {entry.detail}")
    lines.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "11_SYMBOL_INDEX_RAW.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    write_directory_tree()
    write_task_registry_index()
    write_symbol_index()
    print(f"Wrote {OUT_DIR / '02_DIRECTORY_TREE.txt'}")
    print(f"Wrote {OUT_DIR / '04_TASK_REGISTRY_INDEX.md'}")
    print(f"Wrote {OUT_DIR / '11_SYMBOL_INDEX_RAW.txt'}")


if __name__ == "__main__":
    main()
