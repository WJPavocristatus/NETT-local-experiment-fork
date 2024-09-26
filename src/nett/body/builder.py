from gym import Env, Wrapper
from stable_baselines3.common.env_checker import check_env

from nett.body import types
from nett.body.wrappers.dvs import DVSWrapper
# from nett.body import ascii_art

class Body:
    def __init__(self, type: str = "basic",
                    wrappers: list[Wrapper] = [],
                    dvs: bool = False) -> None:
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.type = self._validate_agent_type(type)
        self.wrappers = self._validate_wrappers(wrappers)
        self.dvs = self._validate_dvs(dvs)

    @staticmethod
    def _validate_agent_type(type: str) -> str:
        if type not in types:
            raise ValueError(f"agent type must be one of {types}")
        return type

    @staticmethod
    def _validate_dvs(dvs: bool) -> bool:
        if not isinstance(dvs, bool):
            raise TypeError("dvs should be a boolean [True, False]")
        return dvs

    @staticmethod
    def _validate_wrappers(wrappers: list[Wrapper]) -> list[Wrapper]:
        for wrapper in wrappers:
            if not issubclass(wrapper, Wrapper):
                raise ValueError("Wrappers must inherit from gym.Wrapper")
        return wrappers

    @staticmethod
    def _wrap(env: Env, wrapper: Wrapper) -> Env:
        # wrap env
        env = wrapper(env)
        # check that the env follows Gym API
        env_check = check_env(env, warn=True)
        if env_check != None:
            raise Exception(f"Failed env check")

        return env

    def __call__(self, env: Env) -> Env:
        try:
            # apply DVS wrapper
            if self.dvs:
                env = self._wrap(env, DVSWrapper)
            # apply all custom wrappers
            if self.wrappers:
                for wrapper in self.wrappers:
                    env = self._wrap(env, wrapper)
        except Exception as e:
            self.logger.exception(f"Failed to apply wrappers to environment")
            raise e
        self.env = env
        return self.env
    
    def __enter__(self):
        return self.env

    def __exit__(self):
        self.env.close()
    

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"


    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"


    def _register(self) -> None:
        raise NotImplementedError
