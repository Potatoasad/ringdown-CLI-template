import os 
from tqdm import trange
from typing import Union
from .inferenceobject import InferenceObject
from dataclasses import dataclass
import shutil

@dataclass
class Task:
    name : str
    inference_object : Union[InferenceObject, None]
    
@dataclass
class LazyTask:
    name : str
    path : str
    
    @property 
    def inference_object(self):
        return InferenceObject.from_folder(self.path)

@dataclass
class InjectionTasks:
    foldername : str
    tasks : list
    lazy : Union[bool, None] = None
    no_strains : Union[bool, None] = None

    def is_no_strains(self, no_strains):
        if no_strains is None:
            if self.no_strains is None:
                return True # default is true
            else:
                return self.no_strains
        else:
            return no_strains
        
    def is_lazy(self, lazy):
        if lazy is None:
            if self.lazy is None:
                return True # default is true
            else:
                return self.lazy
        else:
            return lazy
    
    @classmethod
    def from_folder(cls, foldername, lazy=None):
        if lazy is None:
            lazy = False

        _, folders, _ = next(os.walk(foldername))
        tasks = []
        for folder in folders:
            if lazy:
                tasks.append(LazyTask(name=folder, 
                                      path=os.path.join(foldername,folder)))
            else:
                tasks.append(Task(name=folder, 
                                  inference_object=InferenceObject.from_folder(os.path.join(foldername,folder))))
        
        return cls(foldername=foldername, tasks=tasks)

    def task_folder(self, task):
        return os.path.join(self.foldername, task.name)

    def task_to_folder(self, task, no_strains=None):
        no_strains = self.is_no_strains(no_strains)
        task_folder = self.task_folder(task)
        print(f"saving in {task_folder}")
        IO = task.inference_object
        if no_strains:
            IO.fit.result.posterior = IO.fit.result.posterior.drop_vars(['h_det', 'h_det_mode','whitened_residual'])
        IO.to_folder(task_folder)

    def add_task(self, thetask, lazy=None, no_strains=None):
        lazy = self.is_lazy(lazy)
        no_strains = self.is_no_strains(no_strains)
        
        if lazy:
            if not os.path.exists(self.foldername):
                os.mkdir(self.foldername)

            task_folder = os.path.join(self.foldername, thetask.name)
            print(f"saving in {task_folder}")
            IO = thetask.inference_object
            if no_strains:
                if IO.fit.result is not None:
                    IO.fit.result.posterior = IO.fit.result.posterior.drop_vars(['h_det', 'h_det_mode','whitened_residual'])
            IO.to_folder(task_folder)
            self.tasks.append(LazyTask(thetask.name, task_folder))
        else:
            self.tasks.append(thetask)

    def to_folder(self, overwrite=False, no_strains=True):
        if overwrite:
            if os.path.exists(self.foldername):
                shutil.rmtree(self.foldername)
        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)
            
        for i,task in enumerate(self.tasks):
            self.task_to_folder(task)
            
    def run(self, lazy=None, no_strains=None, **kwargs):
        lazy = self.is_lazy(lazy)
        for i in trange(len(self.tasks)):
            IO = self.tasks[i].inference_object
            IO.fit.run(**kwargs)
            if lazy:
                self.task_to_folder(Task(name=self.tasks[i].name,  inference_object=IO), no_strains=no_strains)
