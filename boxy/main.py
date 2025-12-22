from boxy import Boxy
from container import Container
from salloc_model import SallocModel
from vllm import Vllm
from deploy import Deploy

# 1) Construct each collaborator however you like:
allocator = SallocModel(nodes=2, time="00:30:00", partition="debug")
boxy     = Boxy()
container = Container(engine="podman")
vllm     = Vllm(model_name="bert-base-uncased",
                tensor_parallel_size=2,
                max_model_len=4096)

# 2) Inject into Deploy
deployer = Deploy(
    system="hops",
    runtime="podman",
    alloc=allocator,
    boxy=boxy,
    container=container,
    vllm=vllm,
)

# 3) Launch in background
deployer.run(background=True)
print(deployer.status())

# …later…
deployer.kill()
print(deployer.status())
