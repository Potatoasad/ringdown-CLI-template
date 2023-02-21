from ringdb import Database

db = Database("./Data")
db.initialize()

import scripts

scripts.Injections.db_config.db = db

event = db.event("GW150914")

from scripts import *


IT = InjectionTasks.from_folder("mchi_pptest", lazy=True)

IT.run(lazy=True)
