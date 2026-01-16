from boxy.Boxy import Boxy 
from boxy.Container import Container
from boxy.Alloc import SallocModel as Salloc 
from boxy.Vllm import Vllm
from boxy.Deploy import Deploy
import boxy.common as common

# 1) Construct each collaborator however you like:
allocator = Salloc(nodes=2, time="00:30:00", partition="debug")
box = Boxy()
container = Container(runtime="podman")
vllm     = Vllm(model_name=common.models[0],
                tensor_parallel_size=2,
                max_model_len=4096)

# 2) Inject into Deploy
deployer = Deploy(
    system="hops",
    runtime="podman",
    alloc=allocator,
    boxy=box,
    container=container,
    vllm=vllm,
)


# 3) Launch in background
deployer.run(background=False)
print(deployer.status())

# …later…
#deployer.kill()
#print(deployer.status())
