import os
import stat
import logging
from pathlib import Path
# simplify imports
# from nett.brain.builder import Brain
# from nett.body.builder import Body
# from nett.environment.builder import Environment
# from nett.nett import NETT

# from nett.brain import list_encoders, list_algorithms, list_policies


# change permissions of the ml-agents binaries directory

# path to store library cache (such as configs etc)
cache_dir = Path.joinpath(Path.home(), ".cache", "nett") #TODO: See if and how this is used. No mention of it in the codebase.

# set up logging
logging.basicConfig(format="[%(name)s] %(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# path to store ml-agents binaries
for tmp_dir in ["/tmp/ml-agents-binaries", "/tmp/ml-agents-binaries/binaries", "/tmp/ml-agents-binaries/tmp"]:
  if stat.S_IMODE(os.stat(tmp_dir).st_mode) % 0o1000 != 0o777:
    if os.stat(tmp_dir).st_uid == os.getuid() or os.access(tmp_dir, os.W_OK):
      os.chmod(tmp_dir, 0o1777)
    else:
      logger.warning(f"You do not have permission to change the necessary files in '{tmp_dir}'.")