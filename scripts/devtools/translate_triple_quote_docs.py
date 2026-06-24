#!/usr/bin/env python3
from __future__ import annotations

"""Translate standalone triple-double-quoted Python strings from English to Chinese.

The original English string is preserved. A second standalone triple-double-quoted
Chinese string is inserted immediately after it. The script intentionally ignores:

- hash comments;
- assigned strings and strings passed as values;
- triple-single-quoted strings;
- f-strings and dynamically constructed strings.

The translation backend is an NLLB CTranslate2 model. Dependencies and model data can
live outside the repository and are supplied through environment variables or the
command-line arguments below.
"""
"""将独立的 Python 三双引号字符串由英文翻译为中文。

脚本保留原始英文字符串，并在其后立即插入一个独立的三双引号中文字符串。
脚本有意忽略以下内容：

- `#` 注释；
- 赋值字符串以及作为参数值传递的字符串；
- 三单引号字符串；
- f-string 和动态构造的字符串。

翻译后端为 NLLB CTranslate2 模型。依赖项和模型数据可以位于仓库之外，
并通过环境变量或下方的命令行参数提供。
"""

import argparse
import ast
import inspect
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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

SECTION_TRANSLATIONS = {
    "Args:": "参数：",
    "Arguments:": "参数：",
    "Attributes:": "属性：",
    "Example:": "示例：",
    "Examples:": "示例：",
    "Keyword Args:": "关键字参数：",
    "Methods:": "方法：",
    "Note:": "说明：",
    "Notes:": "说明：",
    "Raises:": "异常：",
    "References:": "参考：",
    "Returns:": "返回：",
    "See Also:": "另请参阅：",
    "Warning:": "警告：",
    "Warnings:": "警告：",
    "Yields:": "生成：",
}

EXACT_TRANSLATIONS = {
    "Cleanup for the environment.": "清理环境。",
    "Properties.": "属性。",
    "Operations.": "操作。",
    "Operations - Setup.": "操作 - 初始化。",
    "Operations - MDP.": "操作 - MDP。",
    "Helper functions.": "辅助函数。",
    "Initialize the environment.": "初始化环境。",
    "Load the managers for the environment.": "加载环境管理器。",
    "Get the IO descriptors for the environment.": "获取环境的 IO 描述符。",
    "Export the IO descriptors for the environment.": "导出环境的 IO 描述符。",
    "The physics time-step (in s).": "物理时间步（单位：s）。",
    "The environment stepping time-step (in s).": "环境步进时间步（单位：s）。",
    "This is the lowest time-decimation at which the simulation is happening.": "这是仿真采用的最小时间步。",
    "This is the time-step at which the environment steps forward.": "这是环境向前推进一个步长时使用的时间步。",
    "The environment steps forward in time at a fixed time-step.": "环境按照固定时间步向前推进。",
    "The physics simulation is decimated at a lower time-step.": "物理仿真以更小的时间步运行，并按 decimation 执行。",
    "This is to ensure that the simulation is stable.": "这样可以保证仿真的稳定性。",
    "The device on which the environment is running.": "环境运行所在的设备。",
    "The number of instances of the environment that are running.": "正在运行的环境实例数量。",
    "If a simulation context already exists.": "如果仿真上下文已经存在。",
    "The environment must always create one since it configures the simulation context and controls the simulation.": (
        "环境必须始终自行创建仿真上下文，因为环境需要配置并控制该仿真上下文。"
    ),
    "This function is responsible for creating the various managers (action, observation, events, etc.) for the environment.": (
        "该函数负责为环境创建各种管理器，包括动作管理器、观测管理器和事件管理器等。"
    ),
    "Since the managers require access to physics handles, they can only be created after the simulator is reset (i.e. played for the first time).": (
        "由于这些管理器需要访问物理句柄，因此只能在仿真器完成首次 reset（即第一次开始运行）后创建。"
    ),
    "In case of standalone application (when running simulator from Python), the function is called automatically when the class is initialized.": (
        "在独立运行应用中（即从 Python 启动仿真器时），该函数会在类初始化期间自动调用。"
    ),
    "However, in case of extension mode, the user must call this function manually after the simulator is reset.": (
        "但在 extension 模式下，用户必须在仿真器 reset 后手动调用该函数。"
    ),
    "This is because the simulator is only reset when the user calls :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.": (
        "这是因为只有用户调用 :meth:`SimulationContext.reset_async` 时仿真器才会 reset，而构造函数中无法调用 async 函数。"
    ),
}

# Longest phrases must appear first. Only terms that machine translation commonly
# mistranslates in robotics/RL are forced. Values are restored after inference.
FORCED_GLOSSARY = [
    ("reinforcement learning", "强化学习"),
    ("imitation learning", "模仿学习"),
    ("motion planning", "运动规划"),
    ("domain randomization", "域随机化"),
    ("scene entity", "场景实体"),
    ("rigid body", "刚体"),
    ("base environment", "基础环境"),
    ("Markov Decision Processes", "马尔可夫决策过程"),
    ("end-effector", "末端执行器"),
    ("floating base", "浮动基座"),
    ("regular expressions", "正则表达式"),
    ("regular expression", "正则表达式"),
    ("quaternions", "四元数"),
    ("quaternion", "四元数"),
    ("quadrupeds", "四足机器人"),
    ("quadruped", "四足机器人"),
    ("rendering", "渲染"),
    ("tensors", "张量"),
    ("tensor", "张量"),
    ("tuples", "元组"),
    ("tuple", "元组"),
    ("low-level", "低层"),
    ("time step", "时间步"),
    ("time-step", "时间步"),
    ("managers", "管理器"),
    ("manager", "管理器"),
    ("weighted", "加权"),
    ("simulation", "仿真"),
    ("observations", "观测"),
    ("observation", "观测"),
    ("actions", "动作"),
    ("action", "动作"),
    ("policies", "策略"),
    ("policy", "策略"),
    ("episodes", "回合"),
    ("episode", "回合"),
    ("rollouts", "rollout"),
    ("rollout", "rollout"),
    ("articulations", "关节系统"),
    ("articulation", "关节系统"),
    ("joint torques", "关节力矩"),
    ("joint torque", "关节力矩"),
    ("joint positions", "关节位置"),
    ("joint position", "关节位置"),
    ("joint velocities", "关节速度"),
    ("joint velocity", "关节速度"),
    ("joints", "关节"),
    ("joint", "关节"),
    ("poses", "位姿"),
    ("pose", "位姿"),
    ("instances", "实例"),
    ("instance", "实例"),
    ("properties", "属性"),
    ("property", "属性"),
    ("descriptors", "描述符"),
    ("descriptor", "描述符"),
    ("dictionary", "字典"),
    ("context", "上下文"),
    ("handles", "句柄"),
    ("handle", "句柄"),
    ("constructor", "构造函数"),
    ("standalone", "独立运行"),
    ("extension", "扩展"),
    ("device", "设备"),
    ("decimated", "降采样"),
    ("terms", "项"),
    ("term", "项"),
    ("timeout", "超时"),
    ("time-out", "超时"),
    ("truncated", "截断"),
    ("clipping", "裁剪"),
]

PROPER_TERMS = [
    "Isaac Lab",
    "Isaac Sim",
    "ManagerBasedEnv",
    "ManagerBasedRLEnv",
    "DirectRLEnv",
    "DirectMARLEnv",
    "InteractiveScene",
    "SimulationContext",
    "Gymnasium",
    "PyTorch",
    "TensorDict",
    "NumPy",
    "Hydra",
    "PhysX",
    "Fabric",
    "Omniverse",
    "OpenUSD",
    "RSL-RL",
    "RL-Games",
    "Stable-Baselines3",
    "SKRL",
    "PPO",
    "MDP",
    "MARL",
    "RL",
    "API",
    "GUI",
    "CPU",
    "GPU",
    "CUDA",
    "RTX",
    "USD",
    "URDF",
    "MJCF",
    "ONNX",
    "HDF5",
    "async",
]

POST_REPLACEMENTS = [
    ("基座环境", "基础环境"),
    ("基本环境", "基础环境"),
    ("基于管理者", "基于管理器"),
    ("基于经理", "基于管理器"),
    ("环境管理者", "环境管理器"),
    ("环境经理", "环境管理器"),
    ("奖励经理", "奖励管理器"),
    ("观察经理", "观测管理器"),
    ("管理人员", "管理器"),
    ("管理者", "管理器"),
    ("经理", "管理器"),
    ("报酬", "奖励"),
    ("权重奖励条件", "加权奖励项"),
    ("加权奖励条件", "加权奖励项"),
    ("用户定义的术语", "用户定义项"),
    ("用户定义术语", "用户定义项"),
    ("奖励条款", "奖励项"),
    ("终止条款", "终止项"),
    ("条款", "项"),
    ("行动管理器", "动作管理器"),
    ("操作管理器", "动作管理器"),
    ("行动", "动作"),
    ("原始操作", "原始动作"),
    ("观察", "观测"),
    ("政策", "策略"),
    ("环保", "环境"),
    ("台阶", "步"),
    ("插曲", "回合"),
    ("现场", "场景"),
    ("指挥", "命令"),
    ("活动管理器", "事件管理器"),
    ("活动", "事件"),
    ("物理控件", "物理句柄"),
    ("物理处理", "物理句柄"),
    ("手柄", "句柄"),
    ("联合扭矩", "关节力矩"),
    ("联合位置", "关节位置"),
    ("联合速度", "关节速度"),
    ("终端效应器", "末端执行器"),
    ("最终效应的姿势", "末端执行器位姿"),
    ("移动基地", "移动基座"),
    ("浮动基地", "浮动基座"),
    ("重新设置", "重置"),
    ("录制器", "记录器"),
    ("创造", "创建"),
    ("运作", "运行"),
    ("互动", "交互"),
    ("词典", "字典"),
    ("按键", "键"),
    ("工作流程", "工作流"),
    ("马科夫", "马尔可夫"),
    ("决策流程", "决策过程"),
    ("基层环境", "基础环境"),
    ("特定任务的数量", "任务特定量"),
    ("这些数量", "这些量"),
    ("键作为组名称和值作为IO描述符", "键为组名、值为 IO 描述符"),
    ("摄像头，镜头等", "摄像头、激光雷达等"),
    ("摄像头，罩等", "摄像头、激光雷达等"),
    ("低级命令", "低层命令"),
    ("两者的产量", "两者的乘积"),
    ("对MDP的定义无知", "与 MDP 的具体定义无关"),
    ("这个功能", "该函数"),
    ("该功能", "该函数"),
    ("被毁灭", "进行降采样"),
    ("毁灭", "降采样"),
    ("模拟器", "仿真器"),
    ("模拟", "仿真"),
    ("配置文件类", "配置类"),
    ("增强学习", "强化学习"),
    ("加强学习", "强化学习"),
    ("图书馆", "库"),
    ("管理员", "管理器"),
    ("图普尔", "元组"),
    ("节目", "回合"),
    ("健身房", "Gym"),
    ("常规的表达式", "正则表达式"),
    ("常规表达式", "正则表达式"),
    ("定期表达式", "正则表达式"),
    ("密钥", "键"),
    ("火浮动子", "torch 浮点张量"),
    ("火浮动数", "torch 浮点张量"),
    ("火布尔式子", "torch 布尔张量"),
    ("无国有", "无状态"),
    ("有国有", "有状态"),
    ("超级类", "超类"),
    ("连续化", "序列化"),
    ("消散化", "反序列化"),
    ("环境身份证", "环境 ID"),
    ("在False上默认", "默认为 False"),
    ("在True上默认", "默认为 True"),
    ("在None上默认", "默认为 None"),
    ("默认的None", "默认为 None"),
    ("默认的 None", "默认为 None"),
]

INLINE_PROTECTION_PATTERNS = [
    re.compile(r"https?://[^\s)>]+"),
    re.compile(r"\b(?:i\.e|e\.g)\."),
    re.compile(r":(?:class|attr|meth|mod|func|obj|ref|doc):`[^`]+`"),
    re.compile(r"``[^`]+``"),
    re.compile(r"`[^`]+`"),
    re.compile(r"\{[^{}\n]+\}"),
    re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*"),
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?::[A-Za-z_][A-Za-z0-9_]*)?\b"),
    re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+\b"),
    re.compile(r"\b[a-z]+_[A-Za-z0-9_]+\b"),
    re.compile(r"\b(?:cfg|env|envs|dt|sim|prim|prims|kwargs|args|env_ids|num_envs|None|True|False)\b"),
]


@dataclass
class Target:
    path: Path
    node: ast.Expr
    value: str
    insertion_offset: int
    indent: str
    is_multiline: bool


@dataclass
class RenderPart:
    kind: str
    text: str = ""
    prefix: str = ""
    indent: str = ""


class PlaceholderProtector:
    def __init__(self) -> None:
        self.values: list[str] = []

    def add(self, value: str) -> str:
        placeholder = f"X{len(self.values)}X"
        self.values.append(value)
        return placeholder

    def protect_regex(self, text: str, pattern: re.Pattern[str]) -> str:
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if re.fullmatch(r"X\d+X", value):
                return value
            return self.add(value)

        return pattern.sub(replace, text)

    def restore(self, text: str) -> str:
        for index, value in enumerate(self.values):
            placeholder = f"X{index}X"
            text = re.sub(rf"X\s*{index}\s*X", lambda _match, output=value: output, text, flags=re.IGNORECASE)
        return text

    def all_present(self, text: str) -> bool:
        return all(re.search(rf"X\s*{index}\s*X", text, flags=re.IGNORECASE) for index in range(len(self.values)))


class BatchTranslator:
    def __init__(
        self,
        cache_path: Path,
        model_path: Path | None = None,
        compute_type: str = "int8",
        beam_size: int = 2,
    ) -> None:
        try:
            import ctranslate2
        except ImportError as exc:
            raise RuntimeError(
                "CTranslate2 is unavailable. Set PYTHONPATH to the temporary translation runtime."
            ) from exc

        self.model_path = model_path
        self.backend = "nllb" if model_path is not None else "argos"
        self.beam_size = beam_size
        if self.backend == "nllb":
            try:
                from transformers import AutoTokenizer
                from transformers.utils import logging as transformers_logging
            except ImportError as exc:
                raise RuntimeError("Transformers tokenizer dependencies are unavailable.") from exc
            transformers_logging.set_verbosity_error()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                src_lang="eng_Latn",
                tgt_lang="zho_Hans",
                local_files_only=True,
                use_fast=False,
            )
            ct2_model_path = model_path
        else:
            from argostranslate import package

            packages = [
                pkg for pkg in package.get_installed_packages() if pkg.from_code == "en" and pkg.to_code == "zh"
            ]
            if not packages:
                raise RuntimeError("No installed Argos en->zh package was found.")
            self.pkg = packages[0]
            ct2_model_path = self.pkg.package_path / "model"

        self.translator = ctranslate2.Translator(
            str(ct2_model_path),
            device="cpu",
            inter_threads=max(1, min(4, os.cpu_count() or 1)),
            intra_threads=0,
            compute_type=compute_type,
        )
        self.cache_path = cache_path
        if cache_path.exists():
            self.cache: dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    def translate_all(self, texts: Iterable[str], batch_size: int = 128) -> None:
        all_texts = {text for text in texts if text}
        for text in all_texts:
            if text in EXACT_TRANSLATIONS:
                self.cache[text] = EXACT_TRANSLATIONS[text]
        missing = sorted(text for text in all_texts if text not in self.cache)
        for start in range(0, len(missing), batch_size):
            original_batch = missing[start : start + batch_size]
            protected_batch: list[str] = []
            protectors: list[PlaceholderProtector] = []
            for text in original_batch:
                protected, protector = protect_text(text)
                protected_batch.append(protected)
                protectors.append(protector)

            translated_values = self._translate_raw_batch(protected_batch, batch_size=batch_size)

            for original, protector, value in zip(original_batch, protectors, translated_values):
                if protector.all_present(value):
                    value = protector.restore(value)
                else:
                    value = self._translate_piecewise(original)
                value = polish_translation(value)
                self.cache[original] = value if value else original

            self._save_cache()
            print(f"[translate] {min(start + batch_size, len(missing))}/{len(missing)}", flush=True)

    def get(self, text: str) -> str:
        return self.cache[text]

    def _translate_raw_batch(self, texts: list[str], batch_size: int = 128) -> list[str]:
        if not texts:
            return []
        if self.backend == "nllb":
            tokenized = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text)) for text in texts]
            target_prefix = [["zho_Hans"]] * len(tokenized)
        else:
            tokenized = [self.pkg.tokenizer.encode(text) for text in texts]
            target_prefix = [[self.pkg.target_prefix]] * len(tokenized) if self.pkg.target_prefix else None
        translated = self.translator.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            replace_unknowns=True,
            max_batch_size=batch_size,
            batch_type="tokens",
            beam_size=self.beam_size,
            num_hypotheses=1,
            length_penalty=0.2,
            return_scores=False,
        )
        values: list[str] = []
        for result in translated:
            if self.backend == "nllb":
                token_ids = self.tokenizer.convert_tokens_to_ids(result.hypotheses[0])
                value = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            else:
                value = self.pkg.tokenizer.decode(result.hypotheses[0]).strip()
                if self.pkg.target_prefix and value.startswith(self.pkg.target_prefix):
                    value = value[len(self.pkg.target_prefix) :].lstrip()
            values.append(value)
        return values

    def _translate_piecewise(self, text: str) -> str:
        parts = split_protected_parts(text)
        natural = [value for kind, value in parts if kind == "translate" and re.search(r"[A-Za-z]", value)]
        translations = iter(self._translate_raw_batch(natural, batch_size=32))
        output: list[str] = []
        for kind, value in parts:
            if kind == "literal":
                output.append(value)
            elif re.search(r"[A-Za-z]", value):
                output.append(next(translations))
            else:
                output.append(value)
        return "".join(output)

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.cache_path)


def protect_text(text: str) -> tuple[str, PlaceholderProtector]:
    protector = PlaceholderProtector()

    for pattern in INLINE_PROTECTION_PATTERNS:
        text = protector.protect_regex(text, pattern)

    return text, protector


def split_protected_parts(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[int, int, int, str]] = []
    for priority, pattern in enumerate(INLINE_PROTECTION_PATTERNS):
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), priority, match.group(0)))
    candidates.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))
    selected: list[tuple[int, int, str]] = []
    cursor = -1
    for start, end, _priority, output in candidates:
        if start < cursor:
            continue
        selected.append((start, end, output))
        cursor = end

    parts: list[tuple[str, str]] = []
    cursor = 0
    for start, end, output in selected:
        if start > cursor:
            parts.append(("translate", text[cursor:start]))
        parts.append(("literal", output))
        cursor = end
    if cursor < len(text):
        parts.append(("translate", text[cursor:]))
    return parts


def polish_translation(text: str) -> str:
    for old, new in POST_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"\s+([，。；：！？])", r"\1", text)
    text = text.replace(",", "，").replace(";", "；")
    text = re.sub(r"\.(?=\s|$)", "。", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def iter_python_files(root: Path, selected_paths: list[str]) -> list[Path]:
    paths = [root / item for item in selected_paths] if selected_paths else [root]
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if not candidate.is_file() or is_excluded(candidate.relative_to(root)):
                    continue
                is_python = candidate.suffix == ".py"
                is_python_template = (
                    candidate.suffix == ""
                    and "tools" in candidate.parts
                    and "template" in candidate.parts
                    and '"""' in candidate.read_text(encoding="utf-8", errors="ignore")
                )
                if is_python or is_python_template:
                    files.append(candidate)
    return sorted(set(files))


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    return offsets


def line_end_offset(text: str, offsets: list[int], line_number: int) -> int:
    start = offsets[line_number - 1]
    newline = text.find("\n", start)
    return len(text) if newline == -1 else newline


def has_chinese_translation_after(text: str, insertion_offset: int, indent: str) -> bool:
    tail = text[insertion_offset : insertion_offset + 12000]
    pattern = re.compile(
        rf"^\s*\n{re.escape(indent)}(?:[rRuU])?\"\"\"(?P<body>.*?)(?:\"\"\")",
        flags=re.DOTALL,
    )
    match = pattern.match(tail)
    return bool(match and re.search(r"[\u3400-\u9fff]", match.group("body")))


def collect_targets(path: Path) -> tuple[str, list[Target]]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        print(f"[skip syntax] {path}: {exc}", file=sys.stderr)
        return text, []

    offsets = line_offsets(text)
    targets: list[Target] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
        ):
            continue
        segment = ast.get_source_segment(text, node.value) or ""
        stripped = segment.lstrip()
        prefix_stripped = stripped[1:] if stripped[:1].lower() in {"r", "u"} else stripped
        if not prefix_stripped.startswith('"""'):
            continue
        value = node.value.value
        # Chinese translations intentionally retain technical terms such as MDP,
        # Tensor and API. Never treat those retained terms as a new English target.
        if re.search(r"[\u3400-\u9fff]", value):
            continue
        if not re.search(r"[A-Za-z]", value):
            continue

        insertion_node = node
        # A future import must directly follow the module docstring. For that one
        # legal exception, place the Chinese string after all future imports.
        if tree.body and node is tree.body[0]:
            for statement in tree.body[1:]:
                if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
                    insertion_node = statement
                else:
                    break
        insertion = line_end_offset(text, offsets, insertion_node.end_lineno)
        indent = " " * node.col_offset
        if has_chinese_translation_after(text, insertion, indent):
            continue
        targets.append(
            Target(
                path=path,
                node=node,
                value=value,
                insertion_offset=insertion,
                indent=indent,
                is_multiline="\n" in segment,
            )
        )
    targets.sort(key=lambda target: target.insertion_offset)
    return text, targets


def split_long_text(text: str, max_chars: int = 360) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    abbreviations = {
        "i.e.": "I_DOT_E_DOT",
        "e.g.": "E_DOT_G_DOT",
    }
    protected = text
    for abbreviation, placeholder in abbreviations.items():
        protected = protected.replace(abbreviation, placeholder)
    pieces = re.split(r"(?<=[.!?])\s+", protected)
    pieces = [
        piece.replace("I_DOT_E_DOT", "i.e.").replace("E_DOT_G_DOT", "e.g.") for piece in pieces
    ]
    chunks: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= max_chars:
            chunks.append(piece)
            continue
        subpieces = re.split(r"(?<=[;:])\s+", piece)
        current = ""
        for subpiece in subpieces:
            if current and len(current) + 1 + len(subpiece) > max_chars:
                chunks.append(current)
                current = subpiece
            elif current:
                current += " " + subpiece
            else:
                current = subpiece
        if current:
            chunks.append(current)
    return chunks


def is_code_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return (
        stripped.startswith((">>>", "...", "```", "$ ", "#!"))
        or stripped in {"::", "```python", "```bash", "```text"}
        or bool(re.match(r"^(?:from|import|class|def|return|if|elif|else:|for|while|with|try:|except|raise)\b", stripped))
        or bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+", stripped))
    )


def merge_structural_continuations(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    parameter_pattern = re.compile(
        r"^(\s*[A-Za-z_*][A-Za-z0-9_.*-]*(?:\s+\([^)]*\))?\s*:\s*)(.*)$"
    )
    bullet_pattern = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$")
    while index < len(lines):
        line = lines[index]
        if line.strip() in SECTION_TRANSLATIONS:
            merged.append(line)
            index += 1
            continue
        match = parameter_pattern.match(line) or bullet_pattern.match(line)
        if match is None:
            merged.append(line)
            index += 1
            continue

        base_indent = len(line) - len(line.lstrip())
        continuation: list[str] = []
        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index]
            stripped = candidate.strip()
            candidate_indent = len(candidate) - len(candidate.lstrip())
            if not stripped or candidate_indent <= base_indent:
                break
            if parameter_pattern.match(candidate) or bullet_pattern.match(candidate):
                break
            if is_code_like(candidate) or stripped.startswith((".. ", "```", ">>>")):
                break
            continuation.append(stripped)
            next_index += 1

        if continuation:
            merged.append(line.rstrip() + " " + " ".join(continuation))
            index = next_index
        else:
            merged.append(line)
            index += 1
    return merged


def segment_docstring(value: str) -> list[RenderPart]:
    clean = inspect.cleandoc(value)
    lines = merge_structural_continuations(clean.splitlines())
    parts: list[RenderPart] = []
    paragraph: list[str] = []
    paragraph_indent = ""
    in_fence = False
    code_block_indent: int | None = None
    pending_code_block = False

    def flush_paragraph() -> None:
        nonlocal paragraph, paragraph_indent
        if not paragraph:
            return
        joined = " ".join(item.strip() for item in paragraph)
        chunks = split_long_text(joined)
        for index, chunk in enumerate(chunks):
            parts.append(RenderPart("translate", text=chunk, indent=paragraph_indent if index == 0 else paragraph_indent))
        paragraph = []
        paragraph_indent = ""

    for line in lines:
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if stripped.startswith("```"):
            flush_paragraph()
            in_fence = not in_fence
            parts.append(RenderPart("preserve", text=line))
            continue
        if in_fence:
            flush_paragraph()
            parts.append(RenderPart("preserve", text=line))
            continue

        if code_block_indent is not None:
            if not stripped or len(indent) >= code_block_indent:
                flush_paragraph()
                parts.append(RenderPart("preserve", text=line))
                continue
            code_block_indent = None

        if pending_code_block:
            if not stripped:
                flush_paragraph()
                parts.append(RenderPart("preserve", text=line))
                continue
            if indent:
                code_block_indent = len(indent)
                flush_paragraph()
                parts.append(RenderPart("preserve", text=line))
                pending_code_block = False
                continue
            pending_code_block = False

        if not stripped:
            flush_paragraph()
            parts.append(RenderPart("blank"))
            continue

        if stripped.startswith((".. code-block::", ".. math::", ".. literalinclude::", ".. raw::")):
            flush_paragraph()
            parts.append(RenderPart("preserve", text=line))
            pending_code_block = True
            continue

        if stripped.startswith(".. _"):
            flush_paragraph()
            parts.append(RenderPart("preserve", text=line))
            continue

        if stripped in SECTION_TRANSLATIONS:
            flush_paragraph()
            parts.append(RenderPart("preserve", text=indent + SECTION_TRANSLATIONS[stripped]))
            continue

        directive = re.match(r"^(\s*\.\. (?:note|warning|attention|caution|important|tip)::\s*)(.*)$", line)
        if directive:
            flush_paragraph()
            prefix, body = directive.groups()
            translated_prefix = (
                prefix.replace("note", "说明")
                .replace("warning", "警告")
                .replace("attention", "注意")
                .replace("caution", "谨慎")
                .replace("important", "重要")
                .replace("tip", "提示")
            )
            if body:
                parts.append(RenderPart("translate", text=body, prefix=translated_prefix))
            else:
                parts.append(RenderPart("preserve", text=translated_prefix))
            continue

        if is_code_like(line):
            flush_paragraph()
            parts.append(RenderPart("preserve", text=line))
            continue

        bullet = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$", line)
        if bullet:
            flush_paragraph()
            parts.append(RenderPart("translate", text=bullet.group(2), prefix=bullet.group(1)))
            continue

        rst_field = re.match(r"^(\s*:(?!class:|attr:|meth:|mod:|func:|obj:|ref:|doc:)[^:]+:\s*)(.*)$", line)
        if rst_field:
            flush_paragraph()
            if rst_field.group(2):
                parts.append(RenderPart("translate", text=rst_field.group(2), prefix=rst_field.group(1)))
            else:
                parts.append(RenderPart("preserve", text=line))
            continue

        parameter = re.match(
            r"^(\s*[A-Za-z_*][A-Za-z0-9_.*-]*(?:\s+\([^)]*\))?\s*:\s*)(.*)$",
            line,
        )
        if parameter:
            flush_paragraph()
            if parameter.group(2):
                for index, chunk in enumerate(split_long_text(parameter.group(2))):
                    parts.append(
                        RenderPart(
                            "translate",
                            text=chunk,
                            prefix=parameter.group(1) if index == 0 else " " * len(parameter.group(1)),
                        )
                    )
            else:
                parts.append(RenderPart("preserve", text=line))
            continue

        if not paragraph:
            paragraph_indent = indent
        paragraph.append(stripped)

    flush_paragraph()
    while parts and parts[-1].kind == "blank":
        parts.pop()
    return parts


def collect_translation_units(parts: list[RenderPart]) -> list[str]:
    return [part.text for part in parts if part.kind == "translate" and re.search(r"[A-Za-z]", part.text)]


def wrap_translation(prefix: str, indent: str, text: str, width: int = 100) -> list[str]:
    initial = indent + prefix
    subsequent = indent + (" " * len(prefix))
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [initial]


def render_translation(parts: list[RenderPart], translator: BatchTranslator) -> str:
    output: list[str] = []
    for part in parts:
        if part.kind == "blank":
            output.append("")
        elif part.kind == "preserve":
            output.append(part.text)
        else:
            translated = translator.get(part.text) if re.search(r"[A-Za-z]", part.text) else part.text
            output.extend(wrap_translation(part.prefix, part.indent, translated))
    return "\n".join(output).strip()


def format_chinese_literal(translation: str, indent: str, force_multiline: bool) -> str:
    translation = translation.replace('"""', r"\"\"\"")
    if not force_multiline and "\n" not in translation and len(translation) <= 100:
        return f'{indent}"""{translation}"""'
    lines = translation.splitlines() or [""]
    formatted = [f'{indent}"""{lines[0]}']
    formatted.extend(f"{indent}{line}" if line else "" for line in lines[1:])
    formatted.append(f'{indent}"""')
    return "\n".join(formatted)


def update_file(path: Path, text: str, targets: list[Target], translator: BatchTranslator, write: bool) -> int:
    insertions: list[tuple[int, str]] = []
    for target in targets:
        parts = segment_docstring(target.value)
        translation = render_translation(parts, translator)
        if not translation or not re.search(r"[\u3400-\u9fff]", translation):
            print(f"[skip untranslated] {path}:{target.node.lineno}", file=sys.stderr)
            continue
        literal = format_chinese_literal(translation, target.indent, target.is_multiline)
        insertions.append((target.insertion_offset, "\n" + literal))

    if not insertions:
        return 0

    updated = text
    for offset, insertion in reversed(insertions):
        updated = updated[:offset] + insertion + updated[offset:]

    try:
        compile(updated, str(path), "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"Generated invalid Python for {path}: {exc}") from exc

    if write:
        path.write_text(updated, encoding="utf-8")
    return len(insertions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files or directories relative to --root. Defaults to the whole repo.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/isaaclab-doc-translation-cache.json"))
    parser.add_argument("--model", type=Path, default=None, help="Local NLLB CTranslate2 model directory.")
    parser.add_argument("--beam-size", type=int, default=2, help="Translation beam size. Defaults to 2.")
    parser.add_argument("--write", action="store_true", help="Write translated strings to source files.")
    parser.add_argument("--sample", type=int, default=0, help="Only show the first N translations; do not write.")
    parser.add_argument("--limit-files", type=int, default=0, help="Limit the number of scanned files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    files = iter_python_files(root, args.paths)
    if args.limit_files:
        files = files[: args.limit_files]

    file_targets: list[tuple[Path, str, list[Target]]] = []
    layouts: dict[tuple[Path, int], list[RenderPart]] = {}
    units: list[str] = []
    for path in files:
        text, targets = collect_targets(path)
        if not targets:
            continue
        file_targets.append((path, text, targets))
        for target in targets:
            parts = segment_docstring(target.value)
            layouts[(path, target.node.lineno)] = parts
            units.extend(collect_translation_units(parts))

    print(
        f"[scan] files={len(files)} files_with_targets={len(file_targets)} "
        f"strings={sum(len(item[2]) for item in file_targets)} unique_units={len(set(units))}"
    )
    translator = BatchTranslator(args.cache, model_path=args.model, beam_size=args.beam_size)
    translator.translate_all(units)

    if args.sample:
        shown = 0
        for path, _text, targets in file_targets:
            for target in targets:
                parts = layouts[(path, target.node.lineno)]
                print(f"\n--- {path.relative_to(root)}:{target.node.lineno} ---")
                print("[EN]")
                print(inspect.cleandoc(target.value))
                print("[ZH]")
                print(render_translation(parts, translator))
                shown += 1
                if shown >= args.sample:
                    return
        return

    changed_files = 0
    inserted_strings = 0
    for path, text, targets in file_targets:
        count = update_file(path, text, targets, translator, args.write)
        if count:
            changed_files += 1
            inserted_strings += count
            print(f"[{'write' if args.write else 'check'}] {path.relative_to(root)}: {count}")

    print(
        f"[done] changed_files={changed_files} inserted_strings={inserted_strings} "
        f"mode={'write' if args.write else 'check-only'}"
    )


if __name__ == "__main__":
    main()
