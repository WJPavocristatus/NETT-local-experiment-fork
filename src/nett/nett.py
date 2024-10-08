import subprocess
import time
import shutil
from pathlib import Path
from typing import (
    Any,
    Optional
)
from copy import deepcopy
from itertools import product
from concurrent.futures import (
    ProcessPoolExecutor,
    Future, 
    wait as future_wait,
    FIRST_COMPLETED
)

import pandas as pd
from sb3_contrib import RecurrentPPO
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo
)

from utils.io import mute
from nett.utils.job import Job
from nett.utils.side_channel_logger import Logger
from nett.utils.unity_socket import port_in_use
from utils.analyze import analyze
from brain.builder import Brain
from body.builder import Body
from environment.builder import Environment


class NETT:


    def __init__(
            self,
            brain: Brain,
            body: Body,
            environment: Environment
        ):
      
        logger = Logger()
        self.logger = logger.getChild(__class__.__name__)
        self.brain = brain
        self.body = body
        self.environment = environment        
        # for NVIDIA memory management
        nvmlInit()

    def run(self,
            output_dir: Path | str,
            num_brains: int = 1,
            mode: str = "full",
            train_eps: int = 1000,
            test_eps: int = 20,
            batch_mode: bool = True,
            devices: Optional[list[int]] =  None,
            job_memory: str | int = 4,
            buffer: float = 1.2,
            steps_per_episode: int = 1000,
            conditions: Optional[list[str]] = None,
            verbose: int = True,
            synchronous: bool = False,
            save_checkpoints: bool = False,
            checkpoint_freq: int = 30_000,
            base_port: int = 5004
        ) -> list[Future]:
    
        # set up the output_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {output_dir.resolve()}")

        # calculate iterations
        iterations: dict[str, int] = {}
        if mode in ["train", "full"]:
            iterations["train"] = steps_per_episode * train_eps
        if mode in ["test", "full"]:
            iterations["test"] = test_eps * self.environment.num_test_conditions
            if not issubclass(self.brain.algorithm, RecurrentPPO):
                iterations["test"] *= steps_per_episode

        # initialize job object ## TODO: determin WHY HERE?
        Job.initialize(mode, output_dir, save_checkpoints, steps_per_episode, checkpoint_freq, self.brain.reward, batch_mode, iterations)

        # validate devices
        devices = self._validate_devices(devices)
        self.logger.info(f"Devices that will be used: {devices}")

        # estimate memory for a single job
        if job_memory == "auto":
            self.logger.info("Estimating Job Memory...")
            job_memory = int(buffer * self._estimate_job_memory(devices, base_port))
            self.logger.info(f"Estimated Job Memory: {job_memory / (1024**3)} GiB")
        else:
            job_memory *= buffer * 1024 * 1024 * 1024 # set memory to be in GiB
        

        # get task set
        task_set: set[tuple[str,int]] = self._get_task_set(num_brains, self.environment.imprinting_conditions, conditions)
        
        # schedule jobs
        jobs, waitlist = self._schedule_jobs(task_set, devices, job_memory, base_port, self.logger)
        self.logger.info("Scheduled jobs")

        # launch jobs
        self.logger.info("Launching")
        job_sheet = self._launch_jobs(jobs, synchronous, waitlist, verbose)

        # return control back to the user after launching jobs, do not block
        return job_sheet

    def status(self, job_sheet: dict[Future, Job]) -> pd.DataFrame:
     
        selected_columns = ["brain_id", "condition", "device"]
        filtered_job_sheet = self._filter_job_sheet(job_sheet, selected_columns)
        return pd.json_normalize(filtered_job_sheet)

    # TODO v0.3, make .analyze() a staticmethod so that it does not need a class instance to call
    # TODO v0.3. add support for user specified output_dir
    # Discussion v0.3 is print okay or should we have it log using nett's logger?
    # Discussion v0.3 move this out of the class entirely? from nett import analyze, analyze(...)

    # TODO: find place to insert call for analyze()
    def analyze(
            config: str,
                    run_dir: str | Path,
                    output_dir: Optional[str | Path] = None,
                    ep_bucket: int = 100,
                    num_episodes: int = 1000,
                    bar_order: str | list[int] = "default",
                    color_bars: bool = True) -> None:
    
            # TODO may need to clean up this file structure
            # set paths
            run_dir = Path(run_dir).resolve()
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

            analysis_dir = Path(__file__).resolve().parent.joinpath("analysis")
            if output_dir is None:
                output_dir = run_dir.joinpath("results")
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            chick_data_dir = Path(analysis_dir).joinpath("ChickData", f"{config.lower()}.csv")

            if not chick_data_dir.exists():
                raise ValueError(f"'{config}' is not a valid config.")
            elif not run_dir.exists():
                raise ValueError(f"'{run_dir}' is not a valid run directory.")
            elif not analysis_dir.exists():
                raise ValueError(f"'{analysis_dir}' is not a valid analysis directory. This is likely an error in the package.")

            # translate bar_order for R to read
            bar_order_str = str(bar_order).translate({ord(i): None for i in ' []'}) # remove spaces and brackets from bar_order

            # merge
            print("Running merge")
            subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_merge_csvs.R")),
                            "--logs-dir", str(run_dir),
                            "--results-dir", str(output_dir),
                            "--results-name", "analysis_data",
                            "--csv-train", "train_results.csv",
                            "--csv-test", "test_results.csv"], check=True)

            # train
            print("Running analysis for [train]")
            subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_train_viz.R")),
                            "--data-loc", str(output_dir.joinpath("analysis_data")),
                            "--results-wd", str(output_dir),
                            "--ep-bucket", str(ep_bucket),
                            "--num-episodes", str(num_episodes)], check=True)

            # test
            print("Running analysis for [test]")
            subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_test_viz.R")),
                            "--data-loc", str(output_dir.joinpath("analysis_data")),
                            "--results-wd", str(output_dir),
                            "--bar-order", bar_order_str,
                            "--color-bars", str(color_bars),
                            "--chick-file", str(chick_data_dir)], check=True)

            print(f"Analysis complete. See results at {output_dir}")

    def _execute_job(self, job: Job) -> Future:
        brain: Brain = deepcopy(self.brain)

        # for train
        if job.estimate_memory or job.mode in ["train", "full"]: # TODO: Create a memory estimate method for test
            try:
                # initialize environment with necessary arguments
                with self._wrap_env("train", job.port, job.env_kwargs()) as train_environment:
                    brain.train(train_environment, job)
            except Exception as e:
                self.logger.exception(f"Error in training: {e}")
                raise e

        # for test
        if job.mode in ["test", "full"]:
            try:
                # initialize environment with necessary arguments
                with self._wrap_env("test", job.port, job.env_kwargs()) as test_environment:
                    brain.test(test_environment, job)

                    # if job.estimate_memory:                 
                    #     with open(job.paths["base"] / "mem.txt", "w") as file:
                    #         file.write(str(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(job.device)).used))
            except Exception as e:
                self.logger.exception(f"Error in testing: {e}")
                raise e

        return f"Job Completed Successfully for Brain #{job.brain_id} with Condition: {job.condition}"

    def _estimate_job_memory(self, devices: list[int], base_port: int) -> int:
        self.logger.info("Estimating memory for a single job")
        try:
            # create a temporary directory to hold memory estimate during runtime
            tmp_path = Path("./.tmp/").resolve()
            tmp_path.mkdir(parents=True, exist_ok=True)

            # find the GPU with the most free memory
            free_memory = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device)).free for device in devices]
            most_free_gpu = free_memory.index(max(free_memory))

            # find unused port
            while port_in_use(base_port):
                base_port += 1

            # create a test job to estimate memory
            job = Job(
                brain_id=0, 
                condition=self.environment.imprinting_conditions[0], 
                device=most_free_gpu, 
                index=0,
                port=base_port,
                estimate_memory=True)

            job.save_checkpoints = False

            # change initial port for next job
            base_port += 1

            # calculate current memory usage for baseline for comparison
            pre_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(job.device)).used

            # initializer = mute if not verbose else None
            # executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer)
            executor = ProcessPoolExecutor(max_workers=1, initializer=None)
            job_sheet: dict[Future, dict[str, Job]] = {}

            # run job with estimate_memory set to True
            job_future = executor.submit(self._execute_job, job)
            job_sheet[job_future] = job

            future_wait(job_sheet, return_when=FIRST_COMPLETED)

            with open(Path.joinpath(job.paths["base"], "mem.txt").resolve(), "r") as file:
                post_memory = int(file.readline())
        except Exception as e:
            self.logger.exception(f"Error in estimating memory: {e}")
            raise e
        finally:
            if job.paths["base"].exists():
                shutil.rmtree(job.paths["base"])
        
        # estimate memory allocated
        return post_memory - pre_memory

    @staticmethod
    def _filter_job_sheet(job_sheet: dict[Future, dict[str,Any]], selected_columns: list[str]) -> list[dict[str,bool|str]]:
        # TODO include waitlisted jobs
        runStatus = lambda job_future: {'running': job_future.running()}
        jobInfo = lambda job: {k: getattr(job, k) for k in selected_columns}

        return [runStatus(job_future) | jobInfo(job) for job_future, job in job_sheet.items()]

    @staticmethod
    def _get_memory_status(devices: list[int]) -> dict[int, dict[str, int]]:
        unpack = lambda memory_status: {"free": memory_status.free, "used": memory_status.used, "total": memory_status.total}
        memory_status = {
            device_id : unpack(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_id))) 
            for device_id in devices
        }
        return memory_status
    
    def _launch_jobs(self, jobs: list[Job], wait: bool, waitlist: list[Job], verbose: bool) -> dict[Future, Job]:
     
        try:
            max_workers = 1 if len(jobs) == 1 else None
            initializer = mute if not verbose else None
            executor = ProcessPoolExecutor(max_workers=max_workers, initializer=initializer)
            job_sheet: dict[Future, dict[str, Job]] = {}

            for job in jobs:
                job_future = executor.submit(self._execute_job, job)
                job_sheet[job_future] = job
                time.sleep(1)

            while waitlist:
                done, _ = future_wait(job_sheet, return_when=FIRST_COMPLETED)
                for doneFuture in done:
                    doneJob: Job = job_sheet.pop(doneFuture)
                    freeDevice: int = doneJob.device
                    freePort: int = doneJob.port
                    job = waitlist.pop()
                    job.device = freeDevice
                    job.port = freePort
                    job_future = executor.submit(self._execute_job, job)
                    job_sheet[job_future] = job
                    time.sleep(1)

            if wait:
                while job_sheet:
                    done, _ = future_wait(job_sheet, return_when=FIRST_COMPLETED)
                    for doneFuture in done:
                        job_sheet.pop(doneFuture)
                    time.sleep(1)

            return job_sheet
        except Exception as e:
            print(str(e))

    @staticmethod
    def _get_task_set(num_brains: int, all_conditions: list[str], conditions: Optional[list[str]]) -> set[tuple[str,int]]: #TODO: Create a better name for this method
        # create set of all conditions
        all_conditions_set: set[str] = set(all_conditions)
    
        # check if user-defined their own conditions
        if (conditions is not None):
            # create a set of user-defined conditions
            condition_set: set[str] = set(conditions)

            if not condition_set.issubset(all_conditions_set):
                raise ValueError(f"Unknown conditions: {conditions}. Available conditions are: {all_conditions}")
        # default to all conditions
        else:
            condition_set: set[str] = all_conditions_set

        # create set of all brain-environment combinations
        return set(product(condition_set, set(range(1, num_brains + 1))))

    def _schedule_jobs(self, task_set: set[tuple[str,int]], devices: list[int], job_memory: int, port: int, logger: Logger) -> tuple[list[Job], list[Job]]:
        # create jobs
        jobs: list[Job] = []
        waitlist: list[Job] = []

        # assign devices based on memory availability
        # get the list of devices
        free_devices: list[int] = devices.copy()

        # get the free memory status for each device
        free_device_memory: dict[int, int] = {device: memory_status["free"] for device, memory_status in self._get_memory_status(devices).items()}

        while task_set:
            # if there are no free devices, add jobs to the waitlist
            if not free_devices:
                if not jobs:
                    raise ValueError("No jobs could be scheduled. Job size too large for GPUs. If job_memory='auto', consider setting buffer to 1. Otherwise, consider setting job_memory to a value less than or equal to total free GPU memory / buffer.")
                logger.info("No free devices. Jobs will be queued until a device is available.")
                waitlist = [
                    Job(brain_id, condition, device=-1, index=len(jobs)+i, port=-1) 
                    for i, (condition, brain_id) in enumerate(task_set)
                ]
                logger.warning("Insufficient GPU Memory. Jobs will be queued until memory is available. This may take a while.")
                break

            # remove devices that don't have enough memory
            if free_device_memory[free_devices[-1]] < job_memory:
                logger.info(f"Device {free_devices[-1]} does not have enough memory. Removing from list of available devices.")
                free_devices.pop()
            # assign device to job
            else:
                logger.info(f"Assigning device {free_devices[-1]} to job")
                # create job
                condition, brain_id = task_set.pop()

                # find unused port
                while port_in_use(port):
                    port += 1

                job = Job(
                    brain_id=brain_id, 
                    condition=condition, 
                    device=free_devices[-1], 
                    index=len(jobs),
                    port=port)
                jobs.append(job)

                # change initial port for next job
                port += 1

                # allocate memory
                free_device_memory[free_devices[-1]] -= job_memory
                # rotate devices
                free_devices = [free_devices[-1]] + free_devices[:-1]

        return jobs, waitlist

    @staticmethod
    def _validate_devices(devices: Optional[list[int]]) -> list[int]:
        # check if the devices are available and return the list of devices to be used
        available_devices: list[int] = list(range(nvmlDeviceGetCount()))

        if devices is None:
            devices = available_devices
        elif isinstance(devices, list) and not set(devices).issubset(set(available_devices)):
            raise ValueError("Custom device list lists unknown devices. Available devices are: {available_devices}")

        return devices

    def _wrap_env(self, mode: str, port: int, kwargs: dict[str,Any]) -> Body:
        copy_environment = deepcopy(self.environment)
        copy_environment.initialize(mode, port, **kwargs)
        copy_body = deepcopy(self.body)
        # apply wrappers (body)
        return copy_body(copy_environment)    