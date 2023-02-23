import scripts

from ringdb import Database
db = Database("./Data")
db.initialize()

event = db.event("GW150914")

scripts.Injections.db_config.db = db

from scripts import *

import numpy as np

# Modes
modes = [Mode(n=0), Mode(n=1)]

# params
A_scale = 5e-21
duration = 0.1
target = Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423)
model_name = "mchi"

AP = GaussianAPrior(modes=modes, A_scale=A_scale, flat_A=False)
AP.generate()

KP = KerrBlackHolePrior(M_min=40, M_max=250)

IT = InjectionTasks("blah_2_mode_test", [],lazy=True, no_strains=True)

for i in range(2):
    KI = KerrInjection(params=KP.sample(), modes=modes, polarizations=AP.sample(), noise=DetectorNoise(target=target, strain=event.strain(), seed=np.random.randint(2**32 - 1)))
    IO = InferenceObject(injection=KI, modes=modes, target=target, model=model_name, preconditioning=Preconditioning(ds=4, duration=duration), prior_settings=PriorSettings(A_scale=AP.A_scale, M_min=KP.M_min, M_max=KP.M_max, flat_A=False))
    IT.add_task(Task(f"PPTest-{i}", IO), lazy=True)

IT.run()
