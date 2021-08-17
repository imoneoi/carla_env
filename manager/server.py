import psutil
import subprocess
import json
import os
import signal

import carla
import numpy as np


class ServerManager:
    def __init__(self,
                 global_options: dict,
                 gpu_index: int = 0):
        # default options
        self.options = {
            "path": "../CARLA_0.9.11/CarlaUE4.sh",

            "quality": "Low",

            "resolution_x": 640,
            "resolution_y": 320,

            "retries_on_error": 60,
            "timeout": 5,

            "tm_hybrid_physics_mode": True
        }

        self.options.update(global_options.get("server", {}))

        self.gpu_index = gpu_index
        self.server = None
        self.server_port = None

        self.client = None

        self.tm = None
        self.tm_port = None

        # debug info
        print("Server Options: ", json.dumps(self.options))

    def __del__(self):
        # close client
        if self.tm:
            del self.tm
        if self.client:
            del self.client

        # kill server
        if self.server:
            # kill server process group
            pgid = os.getpgid(self.server.pid)
            os.killpg(pgid, signal.SIGKILL)

    def get(self):
        if self.client is None:
            self.start_server()

            # try connect client
            for _ in range(self.options["retries_on_error"]):
                try:
                    self.client = carla.Client("localhost", self.server_port)
                    self.client.set_timeout(self.options["timeout"])

                    self.client.get_world()

                    break
                except Exception:
                    pass

            # create traffic manager
            self.tm = self.client.get_trafficmanager(self.tm_port)
            self.tm.set_hybrid_physics_mode(self.options["tm_hybrid_physics_mode"])

        return self.client

    def reset(self):
        pass

    @staticmethod
    def is_port_used(port):
        """Checks whether or not a port is used"""
        return port in [conn.laddr.port for conn in psutil.net_connections()]

    def start_server(self):
        # find unused port for server
        self.server_port = 0
        server_port_used = True

        while server_port_used:
            self.server_port = np.random.randint(15000, 32000)
            self.tm_port = self.server_port + 2

            server_port_used = self.is_port_used(self.server_port) or \
                self.is_port_used(self.server_port + 1) or \
                self.is_port_used(self.tm_port)  # TrafficManager port

        # start server
        server_command = [
            self.options["path"],

            "-ResX={}".format(self.options["resolution_x"]),
            "-ResY={}".format(self.options["resolution_y"]),

            "-quality-level={}".format(self.options["quality"]),

            "--carla-rpc-port={}".format(self.server_port)
        ]

        server_env = os.environ.copy()
        server_env.update({
            "CUDA_VISIBLE_DEVICES": str(self.gpu_index)
        })

        server_command = " ".join(server_command)
        print(server_command)
        self.server = subprocess.Popen(
            server_command,
            env=server_env,
            preexec_fn=os.setsid,
            shell=True
        )
