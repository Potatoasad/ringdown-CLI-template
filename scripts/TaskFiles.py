import os
import urllib
import ringdown
import numpy as np
import pandas as pd
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import json


def save_as_gwosc(filepath, data):
    import h5py
    if os.path.exists(filepath):
        os.remove(filepath)
    hf = h5py.File(filepath, 'w')
    meta_group = hf.create_group('meta')
    strain_group = hf.create_group('strain')

    meta_group.create_dataset("GPSstart", data=data.index[0])
    T = data.index[-1]-data.index[0] + (data.index[1]-data.index[0])
    meta_group.create_dataset("Duration", data=T)

    strain_group.create_dataset("Strain", data=data.values)
    
    hf.close()


### Class to download and manage strain files that interfaces with the
### strain file API from the ringdown package
class StrainFileManager:
    def __init__(self, Folder : str, StrainDB):
        self.Folder = Folder
        self.StrainDB = StrainDB
        self.all_events = self.StrainDB.url_df.Event.unique()
        
    def get_url(self,eventname, detector, duration=32.0):
        return self.StrainDB.get_url(eventname, detector, duration=duration)
    
    def filename(self,url):
        return f"{self.Folder}/{url.split('/')[-1]}"
    
    def download_file(self, eventname, detector, duration=32.0, force=False):
        url = self.get_url(eventname, detector, duration)
        file = self.filename(url)
        download_it = False
        if os.path.exists(file):
            download_it = force
        else:
            download_it = True
            
        if download_it:
            print(f"Downloading from {url}")
            filename = self.filename(url)
            print(f"Into folder {filename}")
            urllib.request.urlretrieve(url, filename)
            
        return file
    
    def load_data_dict(self, eventname, ifos=["H1","L1"], duration=32.0):
        data_dict = {}
        for ifo in ifos:
            filepath = self.download_file(eventname, ifo, duration)
            #places = filepath.split("_GWOSC_16KHZ_")
            #postfix = places[-1]
            #prefix = "/".join(places[0].split("/")[0:-1])
            #data_dict[ifo] = prefix + "/{i}-{ifo}_GWOSC_16KHZ_" + postfix
            data_dict[ifo] = filepath
        return data_dict


#SFM = StrainFileManager(alternative_data_folder, db.StrainDB)

class Tasks:
    def __init__(self, tasks, folder):
        self.folder = folder
        self.tasks = tasks
        
    @classmethod
    def from_folder(cls, foldername):
        with open(f'{foldername}/tasks_json','r') as f:
            imported_task_file = json.load(f)
            
        for task in imported_task_file:
            fitt = ringdown.Fit.from_config(task['config'].replace("./",f"{foldername}/"))
            try:
                fitt.result = az.from_netcdf(task['output_file'].replace("./",f"{foldername}/"))
            except FileNotFoundError:
                pass
            task["fit"] = fitt
            
        return cls(imported_task_file, foldername)
    
    def remove_folder(self):
        os.system(f"rm -rf {self.folder}")
    
    def create_task_dir(self, masterfolder=".", overwrite=False):
        script_masterfolder = self.folder
        tasks = self.tasks
        if overwrite:
            self.remove_folder()
        
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        binary_file = f"{script_masterfolder}/ringdown_fit"
        if not os.path.exists(binary_file):
            binary_file_url = "https://raw.githubusercontent.com/Potatoasad/ringdown/main/bin/ringdown_fit"
            urllib.request.urlretrieve(binary_file_url, binary_file)


        for task in tasks:
            run_folder = f"{masterfolder}/{task['name']}"
            script_run_folder = f"{script_masterfolder}/{task['name']}"
            plots_folder = f"{script_masterfolder}/{task['name']}"
            task["folder"] = run_folder
            os.makedirs(script_run_folder, exist_ok = True)

            run_config = f"{masterfolder}/{task['name']}/{task['name']}.cfg"
            script_run_config = f"{script_masterfolder}/{task['name']}/{task['name']}.cfg"
            task["config"] = run_config
            task["fit"].to_config(script_run_config)

            task["output_file"] = run_config.replace(".cfg", "-output.nc")

            task["run_command"] = f"""python {masterfolder}/ringdown_fit {run_config} -o {task["output_file"]}"""

        self.tasks = tasks
        if os.path.exists(f'{script_masterfolder}/taskfile'):
            os.remove(f'{script_masterfolder}/taskfile')

        for task in tasks:
            with open(f'{script_masterfolder}/taskfile', 'a') as taskfile:
                taskfile.write(task["run_command"] + "\n")

        with open(f'{script_masterfolder}/tasks_json','w') as tasks_json:
            json.dump([{k:v for k,v in task.items() if k not in ["fit"]} for task in tasks],
                     tasks_json)



class InjectionTasks:
    def __init__(self, tasks, folder, data=None):
        self.folder = folder
        self.tasks = tasks
        
    @classmethod
    def from_folder(cls, foldername):
        with open(f'{foldername}/tasks_json','r') as f:
            imported_task_file = json.load(f)
            
        for task in imported_task_file:
            os.chdir(f"./{foldername}")
            fitt = ringdown.Fit.from_config(task['config'])
            os.chdir(r"../")
            try:
                fitt.result = az.from_netcdf(task['output_file'].replace("./",f"./{foldername}/"))
            except FileNotFoundError:
                pass
            task["fit"] = fitt
            
            all_data = {}
            for signal_type in task["data_filepaths"].keys():
                all_data[signal_type] = {}
                for ifo in ["H1","L1"]:
                    file_path = task["data_filepaths"][signal_type][ifo].replace("./",f"./{foldername}")
                    all_data[signal_type][ifo] = ringdown.Data.read(path=file_path, kind='gwosc')
                    
            task["data"] = all_data
            task["strain"] = all_data["strain"]
            
        return cls(imported_task_file, foldername)
    
    def remove_folder(self):
        os.system(f"rm -rf {self.folder}")
    
    def create_task_dir(self, masterfolder=".", overwrite=False):
        script_masterfolder = self.folder
        tasks = self.tasks
        if overwrite:
            self.remove_folder()
        
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        binary_file = f"{script_masterfolder}/ringdown_fit"
        if not os.path.exists(binary_file):
            binary_file_url = "https://raw.githubusercontent.com/Potatoasad/ringdown/main/bin/ringdown_fit"
            urllib.request.urlretrieve(binary_file_url, binary_file)

        for task in tasks:
            # Folder for each run
            run_folder = f"{masterfolder}/{task['name']}" # Task Folder relative to the ringdown_fit inside
            script_run_folder = f"{script_masterfolder}/{task['name']}" # Task Folder relative to this code
            plots_folder = f"{script_masterfolder}/{task['name']}"
            task["folder"] = run_folder 
            os.makedirs(script_run_folder, exist_ok = True)
            
            # Add the injection data for each run inside there. 
            # Add the locations of the saved data inside the dictionary
            DataFolder = f"{script_run_folder}/Data"
            run_DataFolder = f"{run_folder}/Data"
            os.makedirs(DataFolder, exist_ok = True)
            
            data_filepaths = {}
            data_filepaths_run = {}
            for signal_type in task["data"].keys():
                data_filepaths[signal_type] = {}
                data_filepaths_run[signal_type] = {}
                for ifo in ["H1","L1"]:
                    file_path = f"{DataFolder}/{signal_type}-{ifo}.h5"
                    save_as_gwosc(file_path, task["data"][signal_type][ifo])
                    data_filepaths[signal_type][ifo] = file_path
                    data_filepaths_run[signal_type][ifo] = f"{run_DataFolder}/{signal_type}-{ifo}.h5"
                    
            # Data is now saved in DataFolder, lets update the fit object to point to those files
            task["fit"].info.update({'data': {'path': data_filepaths_run["strain"], 'kind': 'gwosc'}})
            
            task["data_filepaths"] = data_filepaths
            task["data_filepaths_run"] = data_filepaths_run

            run_config = f"{masterfolder}/{task['name']}/{task['name']}.cfg"
            script_run_config = f"{script_masterfolder}/{task['name']}/{task['name']}.cfg"
            task["config"] = run_config
            task["fit"].to_config(script_run_config)

            task["output_file"] = run_config.replace(".cfg", "-output.nc")

            task["run_command"] = f"""python {masterfolder}/ringdown_fit {run_config} -o {task["output_file"]}"""

        self.tasks = tasks
        if os.path.exists(f'{script_masterfolder}/taskfile'):
            os.remove(f'{script_masterfolder}/taskfile')

        for task in tasks:
            with open(f'{script_masterfolder}/taskfile', 'a') as taskfile:
                taskfile.write(task["run_command"] + "\n")

        with open(f'{script_masterfolder}/tasks_json','w') as tasks_json:
            json.dump([{k:v for k,v in task.items() if k not in ["fit", "data", "strain"]} for task in tasks],
                     tasks_json)


