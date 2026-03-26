import subprocess

class Container:
    """
    Wraps Docker/Podman pull & build operations as subprocesses.

    Example:
        c = Container(runtime="podman")
        c.pull("alpine:latest")
        c.build(
            context_path=".",
            tag="myapp:latest",
            dockerfile="Dockerfile",
            build_args={"VERSION": "1.2.3"},
            no_cache=True
        )
    """

    SUPPORTED_RUNTIMES = ("docker", "podman", "charliecloud")

    def __init__(self, runtime="docker"):
        if runtime not in self.SUPPORTED_RUNTIMES:
            raise ValueError(
                f"Unsupported runtime '{runtime}'. "
                f"Choose one of {self.SUPPORTED_RUNTIMES}"
            )
        self.runtime = runtime

        # TODO unset XDG variables:
        # unset(XDG_RUNTIME_DIR)
        # unset(XDG_SESSION_ID)

        # TODO Hugging face tokens. Something like:
        '''
        if [ -z "${HF_USERNAME}" ]; then
          echo "Error: HF_USERNAME is not set. Please set it."
        exit 1
        fi

        if [ -z "${HF_TOKEN}" ]; then
          echo "Error: HF_TOKEN is not set. Please set it."
         exit 1
        fi
        '''

        # TODO get hostname

        
    def _run(self, args):
        """
        Internal helper to run a subprocess and capture output.
        Raises RuntimeError on failure.
        """
        cmd = [self.runtime] + args
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                #capture_output=True,
                text=True
            )
            return completed.stdout.strip()
        except subprocess.CalledProcessError as e:
            # include stderr in the exception message
            stderr = e.stderr.strip() if e.stderr else "<no stderr>"
            raise RuntimeError(
                f"Command '{' '.join(cmd)}' failed (exit code {e.returncode}):\n{stderr}"
            ) from e

        
    # TODO add pull args
    def pull(self, image):
        """
        Pulls the given image from a registry.

        :param image: e.g. "ubuntu:20.04" or "myregistry/myimage:tag"
        :return: stdout from the pull command
        """
        return self._run(["pull", image])

    
    # TODO add build args
    def build(
        self,
        context_path=".",
        tag=None,
        dockerfile="Dockerfile",
        build_args=None,
        no_cache=False
    ):
        """
        Builds an image from a Dockerfile/Containerfile.

        :param context_path: path to the build context ('.' by default)
        :param tag: name:tag for the built image (required)
        :param dockerfile: path to Dockerfile (default "Dockerfile")
        :param build_args: dict of build-arg key→value
        :param no_cache: if True, pass --no-cache
        :return: stdout from the build command
        """
        if not tag:
            raise ValueError("You must specify a 'tag' for the image.")

        args = ["build", "-t", tag, "-f", dockerfile]

        if no_cache:
            args.append("--no-cache")

        if build_args:
            for k, v in build_args.items():
                args += ["--build-arg", f"{k}={v}"]

        args.append(context_path)

        return self._run(args)

    # TODO add run app in container
    def container_run(self):
        pass
