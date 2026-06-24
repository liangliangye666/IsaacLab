# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script dispatches one or more user-defined Python tasks to workers in a Ray cluster.
Each task, along with its resource requirements and execution parameters, is specified in a YAML configuration file.
Users may define the number of CPUs, GPUs, and the amount of memory to allocate per task via the config file.

Key features:
-------------
- Fine-grained, per-task resource management via config fields (`num_gpus`, `num_cpus`, `memory`).
- Parallel execution of multiple tasks using available resources across the Ray cluster.
- Option to specify node affinity for tasks, e.g., by hostname, node ID, or any node.
- Optional batch (simultaneous) or independent scheduling of tasks.

Task scheduling and distribution are handled via Ray’s built-in resource manager.

YAML configuration fields:
--------------------------
- `pip`: List of extra pip packages to install before running any tasks.
- `py_modules`: List of additional Python module paths (directories or files) to include in the runtime environment.
- `concurrent`: (bool) It determines task dispatch semantics:
    - If `concurrent: true`, **all tasks are scheduled as a batch**. The script waits until
      sufficient resources are available for every task in the batch, then launches all tasks
      together. If resources are insufficient, all tasks remain blocked until the cluster can
      support the full batch.
    - If `concurrent: false`, tasks are launched as soon as resources are available for each
      individual task, and Ray independently schedules them. This may result in non-simultaneous
      task start times.
- `tasks`: List of task specifications, each with:
    - `name`: String identifier for the task.
    - `py_args`: Arguments to the Python interpreter (e.g., script/module, flags, user arguments).
    - `num_gpus`: Number of GPUs to allocate (float or string arithmetic, e.g., "2*2").
    - `num_cpus`: Number of CPUs to allocate (float or string).
    - `memory`: Amount of RAM in bytes (int or string).
    - `node` (optional): Node placement constraints.
        - `specific` (str): Type of node placement, support `hostname`, `node_id`, or `any`.
            - `any`: Place the task on any available node.
            - `hostname`: Place the task on a specific hostname. `hostname` must be specified
              in the node field.
            - `node_id`: Place the task on a specific node ID. `node_id` must be specified in
              the node field.
        - `hostname` (str): Specific hostname to place the task on.
        - `node_id` (str): Specific node ID to place the task on.


Typical usage:
--------------

.. code-block:: bash

    # Print help and argument details:
    python task_runner.py -h

    # Submit tasks defined in a YAML file to the Ray cluster (auto-detects Ray head address):
    python task_runner.py --task_cfg /path/to/tasks.yaml

YAML configuration example-1:
-----------------------------

.. code-block:: yaml

    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    concurrent: false
    tasks:
      - name: "Isaac-Cartpole-v0"
        py_args: "-m torch.distributed.run --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=localhost:29501 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --max_iterations 200 --headless --distributed"
        num_gpus: 2
        num_cpus: 10
        memory: 10737418240
      - name: "script need some dependencies"
        py_args: "script.py --option arg"
        num_gpus: 0
        num_cpus: 1
        memory: 10*1024*1024*1024

YAML configuration example-2:
-----------------------------

.. code-block:: yaml

    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    concurrent: true
    tasks:
    - name: "Isaac-Cartpole-v0-multi-node-train-1"
        py_args: "-m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --max_iterations 1000"
        num_gpus: 1
        num_cpus: 10
        memory: 10*1024*1024*1024
        node:
          specific: "hostname"
          hostname: "xxx"
    - name: "Isaac-Cartpole-v0-multi-node-train-2"
        py_args: "-m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=x.x.x.x:5555 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --max_iterations 1000"
        num_gpus: 1
        num_cpus: 10
        memory: 10*1024*1024*1024
        node:
          specific: "hostname"
          hostname: "xxx"

To stop all tasks early, press Ctrl+C; the script will cancel all running Ray tasks.
"""  # noqa: E501
"""这个脚本将一个或多个用户定义的Python任务发送到Ray集群中的员工。
每个任务及其资源要求和执行参数都在YAML配置文件中指定。
用户可以通过配置文件定义CPUs，GPUs的数量和每个任务分配的内存量。

主要特征:
- 通过配置字段 (`num_gpus`，`num_cpus`，`memory`) 进行细节的每任务资源管理。
- 在 Ray 集群中使用可用的资源进行多项任务的并行执行。
- 选择指定任务的节点亲密性， e.g.，按主机名，节点 ID，或任何节点。
- 选择性 (同时) 或独立的任务安排。

通过Ray的内置资源管理器来处理任务的安排和分配。

YAML配置字段:
- `pip`:在执行任何任务之前需要安装的额外的管包列表。
- `py_modules`:在运行时间环境中包含的额外的Python模块路径 (目录或文件) 列表。
- `concurrent`: (bool) 它确定任务发送语义:
    - 如果 `concurrent: true`，
      **所有任务都被安排为批量**.脚本等待到每个任务中可用的资源足够，然后将所有任务一起启动.如果资源不足，所有任务仍然被阻止，直到集群能够支持整个批量。
    - 如果 `concurrent: false`，任务会在每个单个任务的资源可用时启动，Ray独立安排它们.这可能导致非同时的任务启动时间。
- `tasks`:任务规范列表，每个任务都有:
    - `name`:任务的字符串标识符。
    - `py_args`:用于Python解释器的参数 (e.g.，脚本/模块，旗，用户参数)。
    - `num_gpus`:分配的GPUs数量 (漂浮或字符串算法，e.g.， "2*2")。
    - `num_cpus`:分配的CPUs数量 (漂浮或串)。
    - `memory`:在字节 (int 或字符串) 中 RAM 的数量。
    - `node` (可选):节点放置限制。
        - `specific` (str):节点放置类型，支持`hostname`，`node_id`或`any`。
            - `any`:将任务放置在任何可用的节点上。
            - `hostname`:将任务放置在特定的主机名称上。 `hostname`必须在节点字段中指定。
            - `node_id`:将任务放置在特定的节点 ID。 `node_id`必须在节点字段中指定。
        - `hostname` (str): 具体的主机名称，该任务应用于。
        - `node_id` (str):将任务放置在特定节点 ID 上。


典型使用:

.. code-block:: bash

    # Print help and argument details:
    python task_runner.py -h

    # Submit tasks defined in a YAML file to the Ray cluster (auto-detects Ray head address):
    python task_runner.py --task_cfg /path/to/tasks.yaml

YAML配置示例-1:

.. code-block:: yaml

    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    concurrent: false
    tasks:
      - name: "Isaac-Cartpole-v0"
        py_args: "-m torch.distributed.run --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=localhost:29501 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --max_iterations 200 --headless --distributed"
        num_gpus: 2
        num_cpus: 10
        memory: 10737418240
      - name: "script need some dependencies"
        py_args: "script.py --option arg"
        num_gpus: 0
        num_cpus: 1
        memory: 10*1024*1024*1024

YAML配置示例-2:

.. code-block:: yaml

    pip: ["xxx"]
    py_modules: ["my_package/my_package"]
    concurrent: true
    tasks:
    - name: "Isaac-Cartpole-v0-multi-node-train-1"
        py_args: "-m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --max_iterations 1000"
        num_gpus: 1
        num_cpus: 10
        memory: 10*1024*1024*1024
        node:
          specific: "hostname"
          hostname: "xxx"
    - name: "Isaac-Cartpole-v0-multi-node-train-2"
        py_args: "-m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=x.x.x.x:5555 /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --max_iterations 1000"
        num_gpus: 1
        num_cpus: 10
        memory: 10*1024*1024*1024
        node:
          specific: "hostname"
          hostname: "xxx"

为了早点停止所有任务，按Ctrl+C；脚本将取消所有运行Ray任务。
"""

import argparse
import ast
import operator
from datetime import datetime

import yaml

# Local imports
import util  # isort: skip

# Safe operators for arithmetic expression evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_arithmetic(expr: str) -> int | float:
    """
    Safely evaluate a string containing only arithmetic expressions.

    Supports: +, -, *, /, //, **, % and numeric literals.
    Raises ValueError for any non-arithmetic expressions.

    Args:
        expr: A string containing an arithmetic expression (e.g., "10*1024*1024").

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression contains non-arithmetic operations.
    """
    """安全地评估包含仅算术表达式的字符串。

    Supports: +， -， *， /， //， **， %和数字字体。
    为任何非算术表达式增加ValueError。

    参数：
        expr: 包含算术表达式的字符串 (e.g.， "10*1024*1024")。

    返回：
        这个表达式的数值结果。

    异常：
        ValueError: 如果表达式包含非算术操作。
    """

    def _eval_node(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _SAFE_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            operand = _eval_node(node.operand)
            return _SAFE_OPERATORS[type(node.op)](operand)
        else:
            raise ValueError(f"Unsafe expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return _eval_node(tree)
    except (SyntaxError, TypeError) as e:
        raise ValueError(f"Invalid arithmetic expression: {expr}") from e


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Ray task runner.

    Returns:
        A namespace containing parsed CLI arguments:
            - task_cfg (str): Path to the YAML task file.
            - ray_address (str): Ray cluster address.
            - test (bool): Whether to run a GPU resource isolation sanity check.
    """
    """分析对雷任务运行器的命令行参数。

    返回：
        包含解析的CLI参数的命名空间:
            - task_cfg (str): 进入YAML任务文件的路径。
            - ray_address (str):射线集群地址
            - 测试 (bool):是否进行GPU资源隔离智力检查。
    """
    parser = argparse.ArgumentParser(description="Run tasks from a YAML config file.")
    parser.add_argument("--task_cfg", type=str, required=True, help="Path to the YAML task file.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run nvidia-smi test instead of the arbitrary job,"
            "can use as a sanity check prior to any jobs to check "
            "that GPU resources are correctly isolated."
        ),
    )
    return parser.parse_args()


def parse_task_resource(task: dict) -> util.JobResource:
    """
    Parse task resource requirements from the YAML configuration.

    Args:
        task (dict): Dictionary representing a single task's configuration.
            Keys may include `num_gpus`, `num_cpus`, and `memory`, each either
            as a number or evaluatable string expression.

    Returns:
        util.JobResource: Resource object with the parsed values.
    """
    """从YAML配置解析任务资源要求。

    参数：
        task (dict): 代表单个任务的配置字典。
                     键可能包括`num_gpus`，`num_cpus`和`memory`，每个都作为一个数字或可评估的字符串表达式。

    返回：
        util.JobResource: 有解析值的资源对象。
    """
    resource = util.JobResource()
    if "num_gpus" in task:
        value = task["num_gpus"]
        resource.num_gpus = safe_eval_arithmetic(value) if isinstance(value, str) else value
    if "num_cpus" in task:
        value = task["num_cpus"]
        resource.num_cpus = safe_eval_arithmetic(value) if isinstance(value, str) else value
    if "memory" in task:
        value = task["memory"]
        resource.memory = safe_eval_arithmetic(value) if isinstance(value, str) else value
    return resource


def run_tasks(
    tasks: list[dict], args: argparse.Namespace, runtime_env: dict | None = None, concurrent: bool = False
) -> None:
    """
    Submit tasks to the Ray cluster for execution.

    Args:
        tasks (list[dict]): A list of task configuration dictionaries.
        args (argparse.Namespace): Parsed command-line arguments.
        runtime_env (dict | None): Ray runtime environment configuration containing:
            - pip (list[str] | None): Additional pip packages to install.
            - py_modules (list[str] | None): Python modules to include in the environment.
        concurrent (bool): Whether to launch tasks simultaneously as a batch,
                           or independently as resources become available.

    Returns:
        None
    """
    """提交任务给雷集群执行。

    参数：
        tasks (list[dict]): 任务配置字典列表。
        args (argparse.Namespace): 分析命令行参数。
        runtime_env (dict | None): 射线运行环境配置包含:
            - 现在，我们要做什么?None): 需要安装更多的管包。
            - py_modules (列表[str] None): Python 模块将纳入环境中。
        concurrent (bool): 无论是同时作为一批任务，还是随着资源的开放而独立地启动任务。

    返回：
        None
    """
    job_objs = []
    util.ray_init(ray_address=args.ray_address, runtime_env=runtime_env, log_to_driver=False)
    for task in tasks:
        resource = parse_task_resource(task)
        print(f"[INFO] Creating job {task['name']} with resource={resource}")
        job = util.Job(
            name=task["name"],
            py_args=task["py_args"],
            resources=resource,
            node=util.JobNode(
                specific=task.get("node", {}).get("specific"),
                hostname=task.get("node", {}).get("hostname"),
                node_id=task.get("node", {}).get("node_id"),
            ),
        )
        job_objs.append(job)
    start = datetime.now()
    print(f"[INFO] Creating {len(job_objs)} jobs at {start.strftime('%H:%M:%S.%f')} with runtime env={runtime_env}")
    # submit jobs
    util.submit_wrapped_jobs(
        jobs=job_objs,
        test_mode=args.test,
        concurrent=concurrent,
    )
    end = datetime.now()
    print(
        f"[INFO] All jobs completed at {end.strftime('%H:%M:%S.%f')}, took {(end - start).total_seconds():.2f} seconds."
    )


def main() -> None:
    """
    Main entry point for the Ray task runner script.

    Reads the YAML task configuration file, parses CLI arguments,
    and dispatches tasks to the Ray cluster.

    Returns:
        None
    """
    """雷任务运行程序脚本的主要入口点。

    阅读YAML任务配置文件，解析CLI参数，并将任务发送到Ray集群中。

    返回：
        None
    """
    args = parse_args()
    with open(args.task_cfg) as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]
    runtime_env = {
        "pip": None if not config.get("pip") else config["pip"],
        "py_modules": None if not config.get("py_modules") else config["py_modules"],
    }
    concurrent = config.get("concurrent", False)
    run_tasks(
        tasks=tasks,
        args=args,
        runtime_env=runtime_env,
        concurrent=concurrent,
    )


if __name__ == "__main__":
    main()
