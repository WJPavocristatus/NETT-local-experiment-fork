#Packages for making the environment
import uuid #needed for the communicator
import os #Files and directories

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

# Create the StringLogChannel class. 
class Logger(SideChannel):
    def __init__(self, log_title, log_dir="./EnvLogs/") -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7")) # TODO why this UUID?
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        f_name = os.path.join(log_dir, f"{log_title}.csv")
        self.f = open(f_name, "w")

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.f.write(msg.read_string()) #Write message to log file
        self.f.write("\n") #add new line character

    #This is here because it is required and I currently don't use it.
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
