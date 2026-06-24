# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script dispatches sub-job(s) (individual jobs, use :file:`tuner.py` for tuning jobs)
to worker(s) on GPU-enabled node(s) of a specific cluster as part of an resource-wrapped aggregate
job. If no desired compute resources for each sub-job are specified,
this script creates one worker per available node for each node with GPU(s) in the cluster.
If the desired resources for each sub-job is specified,
the maximum number of workers possible with the desired resources are created for each node
with GPU(s) in the cluster. It is also possible to split available node resources for each node
into the desired number of workers with the ``--num_workers`` flag, to be able to easily
parallelize sub-jobs on multi-GPU nodes. Due to Isaac Lab requiring a GPU,
this ignores all CPU only nodes such as loggers.

Sub-jobs are matched with node(s) in a cluster via the following relation:
sorted_nodes = Node sorted by descending GPUs, then descending CPUs, then descending RAM, then node ID
node_submitted_to = sorted_nodes[job_index % total_node_count]

To check the ordering of sorted nodes, supply the ``--test`` argument and run the script.

Sub-jobs are separated by the + delimiter. The ``--sub_jobs`` argument must be the last
argument supplied to the script.

If there is more than one available worker, and more than one sub-job,
sub-jobs will be executed in parallel. If there are more sub-jobs than workers, sub-jobs will
be dispatched to workers as they become available. There is no limit on the number
of sub-jobs that can be near-simultaneously submitted.

This script is meant to be executed on a Ray cluster head node as an aggregate cluster job.
To submit aggregate cluster jobs such as this script to one or more remote clusters,
see :file:`../submit_isaac_ray_job.py`.

KubeRay clusters on Google GKE can be created with :file:`../launch.py`

Usage:

.. code-block:: bash
    # **Ensure that sub-jobs are separated by the ``+`` delimiter.**
    # Generic Templates-----------------------------------
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py -h
    # No resource isolation; no parallelization:
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py
    --sub_jobs <JOB0>+<JOB1>+<JOB2>
    # Automatic Resource Isolation; Example A: needed for parallelization
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py \
    --num_workers <NUM_TO_DIVIDE_TOTAL_RESOURCES_BY> \
    --sub_jobs <JOB0>+<JOB1>
    # Manual Resource Isolation; Example B:  needed for parallelization
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py --num_cpu_per_worker <CPU> \
    --gpu_per_worker <GPU> --ram_gb_per_worker <RAM> --sub_jobs <JOB0>+<JOB1>
    # Manual Resource Isolation; Example C: Needed for parallelization, for heterogeneous workloads
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py --num_cpu_per_worker <CPU> \
    --gpu_per_worker <GPU1> <GPU2> --ram_gb_per_worker <RAM> --sub_jobs <JOB0>+<JOB1>
    # to see all arguments
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py -h
"""
"""这种脚本将一个特定集群的GPU启用节点中的子工作 (个别工作，用于调整工作的文件:`tuner.py`) 发送给一个工作者作为资源包装的集成工作的一部分。
如果没有指定每个子工作所需的计算资源，则该脚本为每个节点创建一个工作者，每个节点在集群中有GPU(s。
如果每个子工作所需的资源已指定，则每个节点将创建出可用所需资源的最大可能的工人数
with GPU(s) in the cluster. It is also possible to split available node resources for each node
在``--num_workers``标志的工人数量中，可以轻松平行多GPU节点的子工作。
由于艾萨克实验室需要GPU，这忽略了所有CPU只节点，如记录器。

在一个集群中，子工作通过下列关系与节点 (node) 匹配:
sorted_nodes = Node sorted by descending GPUs, then descending CPUs, then descending RAM, then node ID
node_submitted_to = sorted_nodes[job_index % total_node_count]

为了检查排序的节点，输入``--test``参数并运行脚本。

部分工作由+界限器分开。
``--sub_jobs``参数必须是最后一个参数。

如果有多个可用的工人，多个子工作，子工作将同时执行。
如果有比工人多的子工作，
几乎可以同时提交的子工作数量没有限制。

这种脚本是为了在Ray集群头节点执行作为集群工作。
提交像本脚本这样的集群工作到一个或多个远程集群，
see :文件:`../submit_isaac_ray_job.py`。

在Google GKE上可以创建KubeRay集群:文件:`../launch.py`

Usage:

.. code-block:: bash
    # **Ensure that sub-jobs are separated by the ``+`` delimiter.**
    # Generic Templates-----------------------------------
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py -h
    # No resource isolation; no parallelization:
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py
    --sub_jobs <JOB0>+<JOB1>+<JOB2>
    # Automatic Resource Isolation; Example A: needed for parallelization
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py     --num_workers <NUM_TO_DIVIDE_TOTAL_RESOURCES_BY>     --sub_jobs <JOB0>+<JOB1>
    # Manual Resource Isolation; Example B:  needed for parallelization
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py --num_cpu_per_worker <CPU>     --gpu_per_worker <GPU> --ram_gb_per_worker <RAM> --sub_jobs <JOB0>+<JOB1>
    # Manual Resource Isolation; Example C: Needed for parallelization, for heterogeneous workloads
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py --num_cpu_per_worker <CPU>     --gpu_per_worker <GPU1> <GPU2> --ram_gb_per_worker <RAM> --sub_jobs <JOB0>+<JOB1>
    # to see all arguments
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/wrap_resources.py -h
"""

import argparse

import util


def wrap_resources_to_jobs(jobs: list[str], args: argparse.Namespace) -> None:
    """
    Provided a list of jobs, dispatch jobs to one worker per available node,
    unless otherwise specified by resource constraints.

    Args:
        jobs: bash commands to execute on a Ray cluster
        args: The arguments for resource allocation

    """
    """如果提供工作列表，只要资源限制明确其他情况，将工作分配给每一个可用的节点的工人。

    参数：
        jobs: 在Ray集群中执行Bash命令
        args: 资源分配的论点
    """
    job_objs = []
    util.ray_init(
        ray_address=args.ray_address,
        runtime_env={
            "py_modules": None if not args.py_modules else args.py_modules,
        },
        log_to_driver=False,
    )
    gpu_node_resources = util.get_gpu_node_resources(include_id=True, include_gb_ram=True)

    if any([args.gpu_per_worker, args.cpu_per_worker, args.ram_gb_per_worker]) and args.num_workers:
        raise ValueError("Either specify only num_workers or only granular resources(GPU,CPU,RAM_GB).")

    num_nodes = len(gpu_node_resources)
    # Populate arguments
    formatted_node_resources = {
        "gpu_per_worker": [gpu_node_resources[i]["GPU"] for i in range(num_nodes)],
        "cpu_per_worker": [gpu_node_resources[i]["CPU"] for i in range(num_nodes)],
        "ram_gb_per_worker": [gpu_node_resources[i]["ram_gb"] for i in range(num_nodes)],
        "num_workers": args.num_workers,  # By default, 1 worker por node
    }
    args = util.fill_in_missing_resources(args, resources=formatted_node_resources, policy=min)
    print(f"[INFO]: Number of GPU nodes found: {num_nodes}")
    if args.test:
        jobs = ["nvidia-smi"] * num_nodes
    for i, job in enumerate(jobs):
        gpu_node = gpu_node_resources[i % num_nodes]
        print(f"[INFO]: Creating job {i + 1} of {len(jobs)} with job '{job}' to node {gpu_node}")
        print(
            f"[INFO]: Resource parameters: GPU: {args.gpu_per_worker[i]}"
            f" CPU: {args.cpu_per_worker[i]} RAM {args.ram_gb_per_worker[i]}"
        )
        print(f"[INFO] For the node parameters, creating {args.num_workers[i]} workers")
        num_gpus = args.gpu_per_worker[i] / args.num_workers[i]
        num_cpus = args.cpu_per_worker[i] / args.num_workers[i]
        memory = (args.ram_gb_per_worker[i] * 1024**3) / args.num_workers[i]
        job_objs.append(
            util.Job(
                cmd=job,
                name=f"Job-{i + 1}",
                resources=util.JobResource(num_gpus=num_gpus, num_cpus=num_cpus, memory=memory),
                node=util.JobNode(
                    specific="node_id",
                    node_id=gpu_node["id"],
                ),
            )
        )
    # submit jobs
    util.submit_wrapped_jobs(jobs=job_objs, test_mode=args.test, concurrent=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    parser = util.add_resource_arguments(arg_parser=parser)
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
    parser.add_argument(
        "--py_modules",
        type=str,
        nargs="*",
        default=[],
        help=(
            "List of python modules or paths to add before running the job. Example: --py_modules my_package/my_package"
        ),
    )
    parser.add_argument(
        "--sub_jobs",
        type=str,
        nargs=argparse.REMAINDER,
        help="This should be last wrapper argument. Jobs separated by the + delimiter to run on a cluster.",
    )
    args = parser.parse_args()
    if args.sub_jobs is not None:
        jobs = " ".join(args.sub_jobs)
        formatted_jobs = jobs.split("+")
    else:
        formatted_jobs = []
    print(f"[INFO]: Isaac Ray Wrapper received jobs {formatted_jobs=}")
    wrap_resources_to_jobs(jobs=formatted_jobs, args=args)
