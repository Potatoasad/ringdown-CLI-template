from ringdb import Database
import ringdown
import numpy as np

# Set up database
data_folder = "./Data" # Folder where the downloaded strain data is saved

db = Database(data_folder)
db.initialize()

event = db.event("GW150914")

import scripts
scripts.Injections.db_config.db = db
from scripts import *

#######################
# SNR Function
#####################
from tqdm import trange

def SNR_distribution(N, KP, AP):
    snrs = []
    for n in trange(N):
        target = Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423)
        modes = [Mode(n=0), Mode(n=1)]

        CI = KerrInjection(params = KP.sample(),
                              modes = modes,
                              polarizations = AP.sample(),
                              noise = DetectorNoise(target = target, 
                                                    strain = event.strain(), 
                                                    seed=np.random.randint(2**32-1))
                              )
        snrs.append(CI.SNR(t_initial=target.t_geo, duration=0.1)['total'])
        
    return np.array(snrs)


##########################
# CREATE INJECTIONS
##########################
# Modes
modes = [Mode(n=0), Mode(n=1)]

# params
A_scale = 5e-21
duration = 0.1
target = Target(ra=1.95, dec=-1.27, psi=0.82, t_H1=1126259462.423)
model_name = "mchi"

AP = FlatAPrior(modes=modes, A_scale=A_scale)
AP.generate()

KP = KerrBlackHolePrior(M_min=40, M_max=200)

#snrs = SNR_distribution(100, KP, AP)

#print(f" The snr is {np.mean(snrs)} ± {np.std(snrs)}")

IT = InjectionTasks("mchi_pptest", [])

for i in range(2):
    KI = KerrInjection(params = KP.sample(),
                          modes = modes,
                          polarizations = AP.sample(),
                          noise = DetectorNoise(target = target,
                                                strain = event.strain(),
                                                seed = np.random.randint(2**32 - 1)))

    IO = InferenceObject(
        injection = KI,
        modes = modes,
        target = target,
        model = model_name,
        preconditioning = Preconditioning(ds=4, duration=duration),
        prior_settings = PriorSettings(A_scale=AP.A_scale, M_min=KP.M_min, M_max=KP.M_max)
    )

    IT.add_task(Task(f"PPTest-{i}", IO), lazy=True)

#IT = InjectionTasks("mchi_pptest", tasks)

#IT.to_folder(overwrite=False)

#IT2 = IT.from_folder("mchi_pptest-test", lazy=True)

#IT2.run(cores=4)
