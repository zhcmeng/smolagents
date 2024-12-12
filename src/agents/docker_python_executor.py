import sys
import json
import traceback
from pathlib import Path
import docker
import time
import uuid
import signal
from typing import Optional, Dict, Tuple, Any
import subprocess

def read_multiplexed_response(socket):
    """Read and demultiplex all responses from Docker exec socket"""
    socket.settimeout(10.0)

    i = 0
    while True and i<1000:
        # Stream output from socket
        response_data = socket.recv(4096)
        responses = response_data.split(b'\x01\x00\x00\x00\x00\x00')

        # The last non-empty chunk should be our JSON response
        for chunk in reversed(responses):
            if chunk and len(chunk.strip()) > 0:
                try:
                    # Find the start of valid JSON by looking for '{'
                    json_start = chunk.find(b'{')
                    if json_start != -1:
                        decoded = chunk[json_start:].decode('utf-8')
                        result = json.loads(decoded)
                        if "output" in result:
                            return decoded
                except json.JSONDecodeError:
                    continue
        i+=1


class DockerInterpreter:
    def __init__(self, work_dir: Path = Path(".")):
        self.client = docker.from_env()
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self.container = None
        self.exec_id = None
        self.socket = None

    def create_interpreter_script(self) -> str:
        """Create the interpreter script that will run inside the container"""
        script = """
import sys
import code
import json
import traceback
import signal
from threading import Lock

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
                return json.dumps({'output': output, 'more': more, 'error': None}) + '\\n'
            except KeyboardInterrupt:
                return json.dumps({'output': '\\nKeyboardInterrupt\\n', 'more': False, 'error': 'interrupt'}) + '\\n'
            except Exception as e:
                return json.dumps({'output': f"Error: {str(e)}\\n", 'more': False, 'error': str(e)}) + '\\n'

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
        
        # Setup volume mapping
        volumes = {
            str(self.work_dir.resolve()): {"bind": "/workspace", "mode": "rw"}
        }

        for container in self.client.containers.list(all=True):
            if container_name == container.name:
                print(f"Found existing container: {container.name}")
                if container.status != "running":
                    container.start()
                self.container = container
                break
        else: # Create new container
            self.container = self.client.containers.run(
                "python:3.9",
                name=container_name,
                command=["python", "/workspace/interpreter.py"],
                detach=True,
                tty=True,
                stdin_open=True,
                working_dir="/workspace",
                volumes=volumes
            )
            # Install packages in the new container
            print("Installing packages...")
            packages = ["pandas", "numpy"]  # Add your required packages here

            result = self.container.exec_run(
                f"pip install {' '.join(packages)}",
                workdir="/workspace"
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
            tty=True
        )
        
        # Connect to the exec instance
        self.socket = self.client.api.exec_start(
            self.exec_id['Id'],
            socket=True,
            demux=True
        )._sock

    def execute(self, code: str) -> Tuple[str, bool]:
        if not self.container :
            raise Exception("Container not started")
        if not self.socket:
            raise Exception("Socket not started")

        command = json.dumps({'code': code}) + '\n'
        self.socket.send(command.encode())

        response = read_multiplexed_response(self.socket)

        try:
            result = json.loads(response)
            return result['output'], result['more']
        except json.JSONDecodeError:
            return f"Error: Invalid response from interpreter: {response}", False


    def stop(self, remove: bool = False):
        if self.socket:
            try:
                self.socket.close()
            except:
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

def main():
    work_dir = Path("interpreter_workspace")
    interpreter = DockerInterpreter(work_dir)
    
    def signal_handler(signum, frame):
        print("\nExiting...")
        interpreter.stop(remove=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting Python interpreter in Docker...")
    interpreter.start("persistent_python_interpreter2")

    snippet = "import pandas as pd"
    output, more = interpreter.execute(snippet)
    print("OUTPUT1")
    print(output, end='')

    snippet = "pd.DataFrame()"
    output, more = interpreter.execute(snippet)
    print("OUTPUT2")
    print(output, end='')


    print("\nStopping interpreter...")
    interpreter.stop(remove=True)

if __name__ == '__main__':
    main()