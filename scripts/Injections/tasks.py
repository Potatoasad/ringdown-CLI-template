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
    
    @classmethod
    def from_folder(cls, foldername, lazy=False):
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
    
    def to_folder(self, overwrite=False):
        if overwrite:
            if os.path.exists(self.foldername):
                shutil.rmtree(self.foldername)
        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)
            
        for i,task in enumerate(self.tasks):
            name = task.name
            task_folder = os.path.join(self.foldername, name)
            print(f"saving in {task_folder}")
            self.tasks[i].inference_object.to_folder(task_folder)
            
    def run(self):
        for i in trange(len(self.tasks)):
            self.tasks[i].inference_object.fit.run()