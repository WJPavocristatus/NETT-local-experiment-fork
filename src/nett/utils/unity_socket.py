import uuid
import os
import socket
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)


def port_in_use(port) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
    except socket.error:
        return True
    return False

# Create the StringLogChannel class. This is how logging info is communicated between python and unity
class UnitySocket(SideChannel):
    def __init__(self, log_title, log_dir="./EnvLogs/") -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        f_name = os.path.join(log_dir, f"{log_title}.csv")
        self.f = open(f_name, "w")

    #Method from Sidechannel interface.
    def on_message_received(self, msg: IncomingMessage) -> None:
        self.f.write(msg.read_string()) #Write message to log file
        self.f.write("\n") #add new line character

    #This is here because it is required and I currently don"t use it.
    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def log_str(self, msg: str) -> None:
        self.f.write(msg)
        self.f.write("\n")

    def __del__(self) -> None:
        self.f.close()
