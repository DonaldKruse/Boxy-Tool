import subprocess


def sallocModel(nodes=1, time="01:00:00", partition=None, name=None,
                 cluster="cluster", gpus_per_node=None, account=None):

        job_id = None
        slurm_cmd = None

        slurm_cmd = [
            "/usr/bin/salloc",
            f"--nodes={nodes}",
            f"--time={time}",
            #"--parsable",
        ]

        if partition:
            slurm_cmd += [ f"--partition={partition}" ]
        if name:
            slurm_cmd += [ f"--job-name={name}" ]
       # if gpus_per_node:
       #     slurm_cmd += [ f"--gpus-per-node={gpus_per_node}" ]
        if account:
            slurm_cmd += [ f"--account={account}" ]

        out = subprocess.check_output(' '.join(slurm_cmd), stderr=subprocess.STDOUT, text=True, shell=True)
        job_id = out.strip()
        return (job_id, slurm_cmd)

        #except subprocess.CalledProcessError as e:
        #    raise RuntimeError(f"salloc failed (exit {e.returncode}):\n{e.output}, cmd={cmd}") from e



### Test ###

# salloc --job-name vllm --partition short --time 3:00:00 --nodes=1 --account FY140001 --gres=gpu:4
cmd_str = "salloc --job-name vllm --partition short --time 3:00:00 --nodes=1 --account FY140001 --gres=gpu:4"

job_id, slum_cmd = sallocModel( nodes=1, partition="short", name="vllm", gpus_per_node="4", account="FY140001")
