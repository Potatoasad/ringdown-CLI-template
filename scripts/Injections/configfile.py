########################################################
### Config File
import configparser
from numbers import Number

class ConfigFile:
    def __init__(self, filename):
        self.filename = filename
        self.internal = {}
        self.config = configparser.ConfigParser()
    
    def read(self):
        self.config.read(self.filename)
        
    def add_section(self,name):
        try:
            self.config.add_section(name)
        except configparser.DuplicateSectionError:
            pass
        
    def set_val(self, section, key, value, as_str=False):
        self.add_section(section)
        if as_str:
            self.config.set(section,key,"'" + self.stringify(value) + "'")
        else:
            self.config.set(section,key,self.stringify(value))
        
    def write(self):
        with open(self.filename, "w") as config_file:
            self.config.write(config_file)
    
    def stringify(self,the_thing):
        if isinstance(the_thing, list): # If its a list
            return_list = []
            for i in range(len(the_thing)):
                the_str = stringify(the_thing[i])
                return_list.append(the_str)
            return "[" + ", ".join(return_list)   + "]"
        elif isinstance(the_thing, dict): # if its a dict
            return_dict = {}
            for k in the_thing.keys():
                the_str = stringify(the_thing[k])
                return_dict[k] = the_str
            return "dict(" + ", ".join([f"{k} = {v}" for k,v in return_dict.items()]) + ")" 
        elif isinstance(the_thing, str): # if its a string
            return the_thing
        elif isinstance(the_thing, Number): # if its a number
            return str(the_thing)
        else: # Then just create the objects creation string
            return f"{type(the_thing).__name__}(" + ", ".join([f"{k} = {v}" for k,v in the_thing.inputs.items()]) + ")"
        
    def __getitem__(self, *args):
        inputs = args[0]
        if isinstance(inputs, tuple):
            return eval(self.config[inputs[0]][inputs[1]])
        else:
            return self.config.__getitem__(*args)

##########################################################