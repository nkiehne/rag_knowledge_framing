from traitlets.config import Config
import os
import json
import pathlib
import shutil
import subprocess
import shlex
import psutil
import ast
import uuid
from functools import cmp_to_key
import asyncio
from itertools import product
import sys
import warnings
import traceback
import time
import signal

import papermill as pm
import nbformat as nbf
from nbconvert.exporters import PythonExporter

from deploy_utils import IS_DEPLOYED_TASK_FLAG


PORT_START = 29750

LOGDIR_ID = "logdir"
LOGFILE_ID = "logfile"
OVERWRITE_LOGDIR_ID = "overwrite_logdir"
NUM_GPUS_ID = "num_gpus"


def make_param_grid(*args, **kwargs):
    iterables = list(args) + list(kwargs.values())
    return list(product(*iterables))

def export_notebook(path, out_dir=None):
    ''' Exports a jupyter notebook to a Python script 
        The filename will be appened with a ".py" extension.

    '''
    path = pathlib.Path(path).absolute()

    # save to python file
    c = Config()
    e = PythonExporter(c)
    out = e.from_filename(path)
    if out_dir is None:
        out_dir = path.parents[0]
    else:
        out_dir = pathlib.Path(out_dir)
    py_path = out_dir.joinpath(path.with_suffix(".py").name).absolute()
    with open(py_path, "w") as f:
        f.write(out[0])
    return py_path

class BaseTask():

    @classmethod
    def is_runnable(cls, notebook):
        '''
        Returns True if the current class can (theoretically) run the notebook
        '''
        raise NotImplementedError()

    @classmethod
    def _update_params_for_papermill(cls, logdir, config, params):
        ''' 
        Run your class specific logic during papermilling
        a notebook.
        If your derive any arguments needed for cls instantiation,
        make sure to return a corresponding dict!
        '''
        return {}

    @classmethod
    def from_notebook(cls, notebook_path, num_gpus, logdir, overwrite_logdir=False, logfile=None, **params):
        notebook_path = pathlib.Path(notebook_path).absolute()

        # parse config
        config = pm.inspect_notebook(notebook_path)
        # TODO: defaults???

        if num_gpus is None:
            raise ValueError("Missing parameter: num_gpus")

        logdir = pathlib.Path(logdir)
        
        # remove log dir, if instructed and existing
        if overwrite_logdir and logdir.exists():
            shutil.rmtree(logdir)

        # create logdir and parents
        logdir.mkdir(exist_ok=True, parents=True)

        # add logdir settings to the notebook too, to allow other code to log to the same directory:
        if "logdir" in config:
            print(f"Parameter 'logdir' overwritten in notebook to '{logdir}'")
        params["logdir"] = str(logdir) + "/"

        init_kwargs = cls._update_params_for_papermill(logdir, config, params)
        init_kwargs = {} if init_kwargs is None else init_kwargs

        # parameterize notebook with papermill
        expbook_path = notebook_path.parents[0].joinpath(notebook_path.stem + f"_{uuid.uuid4()}.ipynb")

        # temporarily disable ugly warnings that we added new parameters to the notebook..
        
        pm.log.logger.setLevel("ERROR")
        try:
            pm.execute_notebook(
                notebook_path,
                expbook_path,
                prepare_only=True,
                parameters=params,
            )
        finally:
            pm.log.logger.setLevel("WARN")

        # copy notebook, parameterized notebook and python script to logdir for reproducibility :))))
        shutil.copyfile(notebook_path, logdir.joinpath(notebook_path.name))

        shutil.copyfile(expbook_path, logdir.joinpath(expbook_path.name))

        # export python script to logdir
        export_notebook(expbook_path, out_dir=logdir)
        py_path = export_notebook(expbook_path)

        # remove the redundant expbook from original folder
        expbook_path.unlink()

        obj = cls(script_path=py_path, logdir=logdir, num_gpus=num_gpus, logfile=logfile, **init_kwargs)
        return obj
    
    def __init__(self, script_path, logdir, num_gpus=1, logfile=None):
        self.script_path = script_path
        self.logdir = logdir
        self.num_gpus = num_gpus
        self._deployed = asyncio.Future()
        self._task = None
        self.cancelled = False
        self.logfile = logfile
        
    async def await_deployment(self):
        ''' Returns once the task is deployed '''
        await self._deployed
        return

    async def await_done(self):
        await self.await_deployment()
        await self._task
        return
        
    def cancel(self):
        if not self._deployed.done():
            self._deployed.cancel()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self.cancelled = True

    async def run(self, gpus, port):
        raise NotImplementedError()

    def _clean_up(self):
        self.script_path.unlink()

    def _get_default_logfile(self):
        return "out.log"

    def _get_child_proc_env_vars(self, gpus):
        ''' Setup a dict of environment vars that neet to be set for each child process '''
        if isinstance(gpus, int):
            gpus = [gpus]
        gpus = ",".join([str(x) for x in gpus])
        env_vars = {
            "CUDA_VISIBLE_DEVICES":gpus,
            IS_DEPLOYED_TASK_FLAG:True,
        }
        return env_vars

    def deploy(self, gpus, port):
        if self.cancelled:
            raise asyncio.CancelledError()
        if len(gpus) != self.num_gpus:
            raise ValueError(f"Task needs {self.num_gpus} gpus, but {len(gpus)} were given ({gpus})")

        self._task = asyncio.create_task(self.run(gpus, port))
        self._deployed.set_result(True)
        self._task.add_done_callback(lambda x: self._clean_up())
        return self._task


class PythonTask(BaseTask):
    '''
    Executes a python script using python launcher.
    '''

    @classmethod
    def is_runnable(cls, notebook):
        # we can start anything...
        return True

    def _get_default_logfile(self):
        return "python.log"

    async def run(self, gpus=[0], port=27500):
        env_vars = " ".join([f"{k}={v}" for k,v in self._get_child_proc_env_vars(gpus).items()])
        cmd = f"{env_vars} python {self.script_path}"
        print("Starting",cmd)

        # determine name of log file
        if self.logfile is None:
            self.logfile = self._get_default_logfile()
        out = self.logdir.joinpath(self.logfile)

        p = None  # Need to initialize in case something fails earlier
        with open(out, "a") as f:
            try:
                f.write(cmd + "\n")
                f.flush()
                env = os.environ.copy()
                env["PATH"] = f"{sys.exec_prefix}/bin:{env['PATH']}"
                start = time.time()
                p = await asyncio.create_subprocess_shell(cmd, shell=True, stdout=f, stderr=f, env=env, start_new_session=True)
                self.p = p
                await p.wait()
                print("exit wait")
                diff = time.time() - start
                f.write(f"Task finished in {diff}s\n")
                f.flush()
            except asyncio.CancelledError as e:
                print("cancelled, shutting down")
                raise
            except Exception:
                print("Exception in subprocess")
                traceback.print_exc(file=f)
                raise
            finally:
                # ALWAYS clean up the process group if p was started
                if p is not None and p.returncode is None:
                    try:
                        print("Killing process group")
                        os.killpg(p.pid, signal.SIGTERM)
                        #await asyncio.sleep(1)   # Give it a moment
                        #os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Already exited
                    except Exception as exc:
                        print("Exception during process group kill:", exc)
                    try:
                        await p.wait()
                    except Exception:
                        pass


from traitlets.config import Config
import os
import json
import pathlib
import shutil
import subprocess
import shlex
import psutil
import ast
import uuid
from functools import cmp_to_key
import asyncio
from itertools import product
import sys
import warnings
import traceback
import time
import signal

import papermill as pm
import nbformat as nbf
from nbconvert.exporters import PythonExporter

from deploy_utils import IS_DEPLOYED_TASK_FLAG


PORT_START = 29750

LOGDIR_ID = "logdir"
LOGFILE_ID = "logfile"
OVERWRITE_LOGDIR_ID = "overwrite_logdir"
NUM_GPUS_ID = "num_gpus"


def make_param_grid(*args, **kwargs):
    iterables = list(args) + list(kwargs.values())
    return list(product(*iterables))

def export_notebook(path, out_dir=None):
    ''' Exports a jupyter notebook to a Python script 
        The filename will be appened with a ".py" extension.

    '''
    path = pathlib.Path(path).absolute()

    # save to python file
    c = Config()
    e = PythonExporter(c)
    out = e.from_filename(path)
    if out_dir is None:
        out_dir = path.parents[0]
    else:
        out_dir = pathlib.Path(out_dir)
    py_path = out_dir.joinpath(path.with_suffix(".py").name).absolute()
    with open(py_path, "w") as f:
        f.write(out[0])
    return py_path

class BaseTask():

    @classmethod
    def is_runnable(cls, notebook):
        '''
        Returns True if the current class can (theoretically) run the notebook
        '''
        raise NotImplementedError()

    @classmethod
    def _update_params_for_papermill(cls, logdir, config, params):
        ''' 
        Run your class specific logic during papermilling
        a notebook.
        If your derive any arguments needed for cls instantiation,
        make sure to return a corresponding dict!
        '''
        return {}

    @classmethod
    def from_notebook(cls, notebook_path, num_gpus, logdir, overwrite_logdir=False, logfile=None, **params):
        notebook_path = pathlib.Path(notebook_path).absolute()

        # parse config
        config = pm.inspect_notebook(notebook_path)
        # TODO: defaults???

        if num_gpus is None:
            raise ValueError("Missing parameter: num_gpus")

        logdir = pathlib.Path(logdir)
        
        # remove log dir, if instructed and existing
        if overwrite_logdir and logdir.exists():
            shutil.rmtree(logdir)

        # create logdir and parents
        logdir.mkdir(exist_ok=True, parents=True)

        # add logdir settings to the notebook too, to allow other code to log to the same directory:
        if "logdir" in config:
            print(f"Parameter 'logdir' overwritten in notebook to '{logdir}'")
        params["logdir"] = str(logdir) + "/"

        init_kwargs = cls._update_params_for_papermill(logdir, config, params)
        init_kwargs = {} if init_kwargs is None else init_kwargs

        # parameterize notebook with papermill
        expbook_path = notebook_path.parents[0].joinpath(notebook_path.stem + f"_{uuid.uuid4()}.ipynb")

        # temporarily disable ugly warnings that we added new parameters to the notebook..
        
        pm.log.logger.setLevel("ERROR")
        try:
            pm.execute_notebook(
                notebook_path,
                expbook_path,
                prepare_only=True,
                parameters=params,
            )
        finally:
            pm.log.logger.setLevel("WARN")

        # copy notebook, parameterized notebook and python script to logdir for reproducibility :))))
        shutil.copyfile(notebook_path, logdir.joinpath(notebook_path.name))

        shutil.copyfile(expbook_path, logdir.joinpath(expbook_path.name))

        # export python script to logdir
        export_notebook(expbook_path, out_dir=logdir)
        py_path = export_notebook(expbook_path)

        # remove the redundant expbook from original folder
        expbook_path.unlink()

        obj = cls(script_path=py_path, logdir=logdir, num_gpus=num_gpus, logfile=logfile, **init_kwargs)
        return obj
    
    def __init__(self, script_path, logdir, num_gpus=1, logfile=None):
        self.script_path = script_path
        self.logdir = logdir
        self.num_gpus = num_gpus
        self._deployed = asyncio.Future()
        self._task = None
        self.cancelled = False
        self.logfile = logfile
        
    async def await_deployment(self):
        ''' Returns once the task is deployed '''
        await self._deployed
        return

    async def await_done(self):
        await self.await_deployment()
        await self._task
        return
        
    def cancel(self):
        if not self._deployed.done():
            self._deployed.cancel()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self.cancelled = True

    async def run(self, gpus, port):
        raise NotImplementedError()

    def _clean_up(self):
        self.script_path.unlink()

    def _get_default_logfile(self):
        return "out.log"

    def _get_child_proc_env_vars(self, gpus):
        ''' Setup a dict of environment vars that neet to be set for each child process '''
        if isinstance(gpus, int):
            gpus = [gpus]
        gpus = ",".join([str(x) for x in gpus])
        env_vars = {
            "CUDA_VISIBLE_DEVICES":gpus,
            IS_DEPLOYED_TASK_FLAG:True,
        }
        return env_vars

    def deploy(self, gpus, port):
        if self.cancelled:
            raise asyncio.CancelledError()
        if len(gpus) != self.num_gpus:
            raise ValueError(f"Task needs {self.num_gpus} gpus, but {len(gpus)} were given ({gpus})")

        self._task = asyncio.create_task(self.run(gpus, port))
        self._deployed.set_result(True)
        self._task.add_done_callback(lambda x: self._clean_up())
        return self._task


async def is_job_finished(jobid):
    """Check if the job with the specified jobid is still present in the queue."""
    proc = await asyncio.create_subprocess_exec("squeue", "-j", str(jobid), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    # If only header is present, the job is gone
    lines = stdout.decode().strip().splitlines()
    return len(lines) <= 1

async def wait_for_job(jobid, poll_interval=10):
    """Asynchronously wait for a Slurm job to finish."""
    while True:
        finished = await is_job_finished(jobid)
        if finished:
            print(f"Job {jobid} finished.")
            return
        await asyncio.sleep(poll_interval)


class SlurmPythonTask(BaseTask):
    '''
    Executes a python script using slurm launcher.
    '''
    def __init__(self, script_path, logdir, num_gpus=1, logfile=None):
        super().__init__(script_path=script_path, logdir=logdir, num_gpus=0, logfile=logfile)
        self.job_gpus = num_gpus

    @classmethod
    def is_runnable(cls, notebook):
        # we can start anything...
        return True

    def _get_default_logfile(self):
        return "python.log"

    def deploy(self, gpus, port):
        if self.cancelled:
            raise asyncio.CancelledError()
        self._task = asyncio.create_task(self.run(gpus, port))
        self._deployed.set_result(True)
        self._task.add_done_callback(lambda x: self._clean_up())
        return self._task


    async def cancel_slurm_job(self, jobid):
        if jobid is not None:
            print(f"Attempting to cancel slurm job {jobid}")
            try:
                scancel_proc = await asyncio.create_subprocess_exec(
                    "scancel", jobid,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await scancel_proc.communicate()
                if scancel_proc.returncode == 0:
                    print(f"scancel sent for job {jobid}")
                    # Optionally, wait for job to terminate
                    try:
                        await wait_for_job(jobid, poll_interval=1)
                        print(f"Job {jobid} terminated")
                    except Exception as wfj_exc:
                        print(f"Warning: waited for job to finish, but error: {wfj_exc}")
                else:
                    print(f"scancel failed: {stderr.decode().strip()}")
            except Exception as exc:
                print(f"Exception while trying to scancel job: {exc}")
        else:
            print("No jobid found, cannot cancel job! You may need to check the job list manually.")


    
    async def run(self, gpus=[0], port=27500):
        # determine name of log file
        if self.logfile is None:
            self.logfile = self._get_default_logfile()
        out = self.logdir.joinpath(self.logfile)

        cmd = [
            "sbatch",
            "-N", "1",
            "-c", "8",
            "-t", "2-00:00:00",
            "-C", "inet",
            "-p", os.environ.get("PARTITIONS"),
            f"--gres=gpu:{self.job_gpus}",
            f"--output={out}",
            "run_slurm_python_task.sh",
            str(self.script_path)
        ]
        print("Starting",cmd)
        jobid = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True
            )
            self.p = process
            # this never blocks, since the comand just submits the job
            await self.p.wait()
            # instead, we have to check whether the slurm job has finished
            # try get the jobid:
            try:
                stdout, stderr = await process.communicate()
                stdout_str = stdout.decode().strip()
                jobid = stdout_str.split(" ")[-1]
            except Exception as e:
                print("FAILED TO GET JOBID -- GOOD LUCK!")
                raise e
            await wait_for_job(jobid, poll_interval=1)
        except asyncio.CancelledError as e:
            print("cancelled, shutting down")
            # well, now we need to send scancel & wait for jobid to finish
            await self.cancel_slurm_job(jobid)
            raise


class NotebookDeployer():
    def __init__(self, num_gpus=None, gpu_list=None, port_range_start=27500, default_backend="python"):
        '''
        Use num_gpus to specify number of gpus to automatically allocate or
        gpu_list = [0,1,5, ...] to give a list of gpu indices to use
        See `enqueue` for default_backend options.
        '''
        if (num_gpus is None and gpu_list is None):
            raise ValueError("Need either num_gpus or gpu_list as not None arguments")

        self.default_backend = default_backend
        self.num_gpus = num_gpus if gpu_list is None else len(gpu_list)
        self.gpu_list = list(range(num_gpus)) if gpu_list is None else gpu_list
        
        self.tasks = []
        self.finished = []
        self.available = set(self.gpu_list)
        self._portmap = {i:port_range_start+i for i in self.available}
        self._task2slot = {}
        
        self.in_use = set()
        self.pending = []

    def stop_all(self):
        self.pending.clear()
        for t in self.tasks:
            t.cancel()

    def _task_done(self, task):
        # free slots
        print("task done fired")
        if task in self._task2slot:
            self.available.update(self._task2slot[task])
        self.tasks.remove(task)
        self.finished.append(task)
        # check for new deployments
        self._deploy()

    def _deploy(self):
        # check, whether any pending futures could be deployed
        to_remove = []
        for f in self.pending:
            if len(self.available) < f.num_gpus:
                continue
            if f.cancelled:
                to_remove.append(f)
                continue
            gpus = []
            port = None
            if f.num_gpus > 0:
                gpus = [self.available.pop() for _ in range(f.num_gpus)]
                port = self._portmap[gpus[0]]
                self._task2slot[t] = gpus
            
            t = f.deploy(gpus, port)    
            t.add_done_callback(self._task_done)
            self.tasks.append(t)
            to_remove.append(f)

        # remove all started tasks from pending
        for f in to_remove:
            self.pending.remove(f)
    
    def status(self):
        print("pending", len(self.pending))
        print("tasks", len(self.tasks))
        print("finished", len(self.finished))        
        
    def enqueue(self, task=None, notebook=None, backend="default", **notebook_params):
        '''
        Enqueue a BaseTask for deployment.
        `task`: A specific BaseTask object instantiated by you.
        `backend`: which launcher to use. Choose from "python" or "python-slurm" or "default".
            Defaults to "default", which is set in constructor.
        `notebook`: Instead of an explicit task, you can also specify a notebook path, which
            will be converted accordingly.
        `notebook_params`: Optional keywords to pass to papermill during notebook conversion.
        '''

        if (task is None and notebook is None) or (not task is None and not notebook is None): 
            raise ValueError("Specify either `task` or `notebook` (but not both)")

        if notebook is not None:
            if backend == "default":
                backend = self.default_backend

            if backend == "python-slurm":
                task = SlurmPythonTask.from_notebook(notebook, **notebook_params)
            elif backend == "python":
                task = PythonTask.from_notebook(notebook, **notebook_params)
            else:
                raise ValueError("supported values for `backend` are 'python-slurm' and 'python'")

        if task.num_gpus > self.num_gpus:
            raise ValueError(f"Too many gpus requested ({task.num_gpus}), only got {self.num_gpus}")
        self.pending.append(task)
        self._deploy()
        return task