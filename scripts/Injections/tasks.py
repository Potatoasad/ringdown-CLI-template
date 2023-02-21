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

    def is_lazy(self, lazy):
        if lazy is None:
            if self.lazy is None:
                return False
            else:
                return self.lazy
        else:
            return lazy
    
    @classmethod
    def from_folder(cls, foldername, lazy=None):
        lazy = self.is_lazy(lazy)
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

    def task_to_folder(self, task):
        task_folder = self.task_folder(task)
        print(f"saving in {task_folder}")
        task.inference_object.to_folder(task_folder)

    def add_task(self, thetask, lazy=None):
        lazy = self.is_lazy(lazy)
        if lazy:
            if not os.path.exists(self.foldername):
                os.mkdir(self.foldername)

            task_folder = os.path.join(self.foldername, thetask.name)
            print(f"saving in {task_folder}")
            thetask.inference_object.to_folder(task_folder)
            self.tasks.append(LazyTask(thetask.name, task_folder))
        else:
            self.tasks.append(thetask)

    def to_folder(self, overwrite=False):
        if overwrite:
            if os.path.exists(self.foldername):
                shutil.rmtree(self.foldername)
        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)
            
        for i,task in enumerate(self.tasks):
            self.task_to_folder(task)
            
    def run(self, lazy=None):
        lazy = self.is_lazy(lazy)
        for i in trange(len(self.tasks)):
            IO = self.tasks[i].inference_object
            IO.fit.run()
            if lazy:
                self.task_to_folder(Task(name=self.tasks[i].name,  inference_object=IO))
