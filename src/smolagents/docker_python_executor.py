import json
from pathlib import Path
import docker
import time
import uuid
import pickle
import re
from typing import Optional, Dict, Tuple, Set, Any
import types
from .default_tools import BASE_PYTHON_TOOLS


class StateManager:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.state_file = work_dir / "interpreter_state.pickle"
        self.imports_file = work_dir / "imports.txt"
        self.import_pattern = re.compile(r"^(?:from\s+[\w.]+\s+)?import\s+.+$")
        self.imports: Set[str] = set()

    def is_import_statement(self, code: str) -> bool:
        """Check if a line of code is an import statement."""
        return bool(self.import_pattern.match(code.strip()))

    def track_imports(self, code: str):
        """Track import statements for later use."""
        for line in code.split("\n"):
            if self.is_import_statement(line.strip()):
                self.imports.add(line.strip())

    def save_state(self, locals_dict: Dict[str, Any], executor: str):
        """
        Save the current state of variables and imports.

        Args:
            locals_dict: Dictionary of local variables
            executor: 'docker' or 'local' to indicate source
        """
        # Filter out modules, functions, and special variables
        state_dict = {
            "variables": {
                k: v
                for k, v in locals_dict.items()
                if not (
                    k.startswith("_")
                    or callable(v)
                    or isinstance(v, type)
                    or isinstance(v, types.ModuleType)
                )
            },
            "imports": list(self.imports),
            "source": executor,
        }

        with open(self.state_file, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, executor: str) -> Dict[str, Any]:
        """
        Load the saved state and handle imports.

        Args:
            executor: 'docker' or 'local' to indicate destination

        Returns:
            Dictionary of variables to restore
        """
        if not self.state_file.exists():
            return {}

        with open(self.state_file, "rb") as f:
            state_dict = pickle.load(f)

        # First handle imports
        for import_stmt in state_dict["imports"]:
            exec(import_stmt, globals())

        return state_dict["variables"]


def read_multiplexed_response(socket):
    """Read and demultiplex all responses from Docker exec socket"""
    socket.settimeout(10.0)

    i = 0
    while True and i < 1000:
        # Stream output from socket
        response_data = socket.recv(4096)
        responses = response_data.split(b"\x01\x00\x00\x00\x00\x00")

        # The last non-empty chunk should be our JSON response
        if len(responses) > 0:
            for chunk in reversed(responses):
                if chunk and len(chunk.strip()) > 0:
                    try:
                        # Find the start of valid JSON by looking for '{'
                        json_start = chunk.find(b"{")
                        if json_start != -1:
                            decoded = chunk[json_start:].decode("utf-8")
                            result = json.loads(decoded)
                            if "output" in result:
                                return decoded
                    except json.JSONDecodeError:
                        continue
        i += 1


class DockerPythonInterpreter:
    def __init__(self, work_dir: Path = Path(".")):
        self.client = docker.from_env()
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self.container = None
        self.exec_id = None
        self.socket = None
        self.state_manager = StateManager(work_dir)

    def create_interpreter_script(self) -> str:
        """Create the interpreter script that will run inside the container"""
        script = """
import sys
import code
import json
import traceback
import signal
import types
from threading import Lock
import pickle

class PersistentInterpreter(code.InteractiveInterpreter):
    def __init__(self):
        self.locals_dict = {'__name__': '__console__', '__doc__': None}
        super().__init__(self.locals_dict)
        self.lock = Lock()
        self.output_buffer = []
        
    def write(self, data):
        self.output_buffer.append(data)
        
    def run_command(self, source):
        with self.lock:
            self.output_buffer = []
            pickle_path = self.work_dir / "locals.pickle"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    locals_dict_update = pickle.load(f)['variables']
            self.locals_dict.update(locals_dict_update)

            try:
                more = self.runsource(source)
                output = ''.join(self.output_buffer)
                
                if not more and not output and source.strip():
                    try:
                        result = eval(source, self.locals_dict)
                        if result is not None:
                            output = repr(result) + '\\n'
                    except:
                        pass
                output = json.dumps({'output': output, 'more': more, 'error': None}) + '\\n'
            except KeyboardInterrupt:
                output = json.dumps({'output': '\\nKeyboardInterrupt\\n', 'more': False, 'error': 'interrupt'}) + '\\n'
            except Exception as e:
                output = json.dumps({'output': f"Error: {str(e)}\\n", 'more': False, 'error': str(e)}) + '\\n'
            finally:
                with open('/workspace/locals.pickle', 'wb') as f:
                    filtered_locals = {
                        k: v for k, v in self.locals_dict.items() 
                        if not (
                            k.startswith('_')
                            or k in {'pickle', 'f'}
                            or callable(v)
                            or isinstance(v, type)
                            or isinstance(v, types.ModuleType)
                        )
                    }
                    pickle.dump(filtered_locals, f)
            return output

def main():
    interpreter = PersistentInterpreter()
    # Make sure interrupts are handled
    signal.signal(signal.SIGINT, signal.default_int_handler)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            try:
                command = json.loads(line)
                result = interpreter.run_command(command['code'])
                sys.stdout.write(result)
                sys.stdout.flush()
            except json.JSONDecodeError:
                sys.stdout.write(json.dumps({'output': 'Invalid command\\n', 'more': False, 'error': 'invalid_json'}) + '\\n')
                sys.stdout.flush()
        except KeyboardInterrupt:
            sys.stdout.write(json.dumps({'output': '\\nKeyboardInterrupt\\n', 'more': False, 'error': 'interrupt'}) + '\\n')
            sys.stdout.flush()
            continue
        except Exception as e:
            sys.stderr.write(f"Fatal error: {str(e)}\\n")
            break

if __name__ == '__main__':
    main()
"""
        script_path = self.work_dir / "interpreter.py"
        with open(script_path, "w") as f:
            f.write(script)
        return str(script_path)

    def wait_for_ready(self, container: Any, timeout: int = 60) -> bool:
        elapsed_time = 0
        while elapsed_time < timeout:
            try:
                container.reload()
                if container.status == "running":
                    return True
                time.sleep(0.2)
                elapsed_time += 0.2
            except docker.errors.NotFound:
                return False
        return False

    def start(self, container_name: Optional[str] = None):
        if container_name is None:
            container_name = f"python-interpreter-{uuid.uuid4().hex[:8]}"

        self.create_interpreter_script()

        # Setup volume mapping
        volumes = {str(self.work_dir.resolve()): {"bind": "/workspace", "mode": "rw"}}

        for container in self.client.containers.list(all=True):
            if container_name == container.name:
                print(f"Found existing container: {container.name}")
                if container.status != "running":
                    container.start()
                self.container = container
                break
        else:  # Create new container
            self.container = self.client.containers.run(
                "python:3.9",
                name=container_name,
                command=["python", "/workspace/interpreter.py"],
                detach=True,
                tty=True,
                stdin_open=True,
                working_dir="/workspace",
                volumes=volumes,
            )
            # Install packages in the new container
            print("Installing packages...")
            packages = ["pandas", "numpy", "pickle5"]  # Add your required packages here

            result = self.container.exec_run(
                f"pip install {' '.join(packages)}", workdir="/workspace"
            )
            if result.exit_code != 0:
                print(f"Warning: Failed to install: {result.output.decode()}")
            else:
                print(f"Installed {packages}.")

        if not self.wait_for_ready(self.container):
            raise Exception("Failed to start container")

        # Start a persistent exec instance
        self.exec_id = self.client.api.exec_create(
            self.container.id,
            ["python", "/workspace/interpreter.py"],
            stdin=True,
            stdout=True,
            stderr=True,
            tty=True,
        )

        # Connect to the exec instance
        self.socket = self.client.api.exec_start(
            self.exec_id["Id"], socket=True, demux=True
        )._sock

    def _raw_execute(self, code: str) -> Tuple[str, bool]:
        """
        Execute code directly without state management.
        This is the original execute method functionality.
        """
        if not self.container:
            raise Exception("Container not started")
        if not self.socket:
            raise Exception("Socket not started")

        command = json.dumps({"code": code}) + "\n"
        self.socket.send(command.encode())

        response = read_multiplexed_response(self.socket)

        try:
            result = json.loads(response)
            return result["output"], result["more"]
        except json.JSONDecodeError:
            return f"Error: Invalid response from interpreter: {response}", False

    def get_locals_dict(self) -> Dict[str, Any]:
        """Get the current locals dictionary from the interpreter by pickling directly from Docker."""
        pickle_path = self.work_dir / "locals.pickle"
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                try:
                    return pickle.load(f)
                except Exception as e:
                    print(f"Error loading pickled locals: {e}")
                    return {}
        return {}

    def execute(self, code: str) -> Tuple[str, bool]:
        # Track imports before execution
        self.state_manager.track_imports(code)

        output, more = self._raw_execute(code)

        # Save state after execution
        self.state_manager.save_state(self.get_locals_dict(), "docker")
        return output, more

    def stop(self, remove: bool = False):
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass

        if self.container:
            try:
                self.container.stop()
                if remove:
                    self.container.remove()
                self.container = None
            except docker.errors.APIError as e:
                print(f"Error stopping container: {e}")
                raise


def execute_locally(code: str, work_dir: Path, tools: Dict[str, Any]) -> Any:
    from .local_python_executor import evaluate_python_code, BASE_BUILTIN_MODULES

    """Execute code locally with state transfer."""
    state_manager = StateManager(work_dir)

    # Track imports
    state_manager.track_imports(code)

    # Load state from Docker if available
    locals_dict = state_manager.load_state("local")

    # Execute in a new namespace with loaded state
    namespace = {}
    namespace.update(locals_dict)

    output = evaluate_python_code(
        code,
        tools,
        {},
        namespace,
        BASE_BUILTIN_MODULES,
    )

    # Save state for Docker
    state_manager.save_state(namespace, "local")
    return output


def create_tools_regex(tool_names):
    # Escape any special regex characters in tool names
    escaped_names = [re.escape(name) for name in tool_names]
    # Join with | and add word boundaries
    pattern = r"\b(" + "|".join(escaped_names) + r")\b"
    return re.compile(pattern)


def execute_code(code: str, tools: Dict[str, Any], work_dir: Path, interpreter):
    """Execute code with automatic switching between Docker and local."""
    lines = code.split("\n")
    current_block = []
    tool_regex = create_tools_regex(
        list(tools.keys()) + ["print"]
    )  # Added print for testing

    tools = {
        **BASE_PYTHON_TOOLS.copy(),
        **tools,
    }

    for line in lines:
        if tool_regex.search(line):
            # Execute accumulated Docker code if any
            if current_block:
                output, more = interpreter.execute("\n".join(current_block))
                print(output, end="")
                current_block = []

            output = execute_locally(line, work_dir, tools)
            if output:
                print(output, end="")
        else:
            current_block.append(line)

    # Execute any remaining Docker code
    if current_block:
        output, more = interpreter.execute("\n".join(current_block))
        print(output, end="")


__all__ = ["DockerPythonInterpreter", "execute_code"]
