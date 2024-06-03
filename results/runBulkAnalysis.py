import os
import time
from multiprocessing import Pool


types = [
    # "./difuse",
    # "./difuseWide",
    # "./plastic",
    # "./plasticNarrow",
    # "./metal",
    # "./metalNarrow",
    "./armadillo_difuse",
    "./armadillo_plastic",
]


start = time.time()
pool = Pool(processes=4)
#pool.map(print, ["python3 analyseBulk.py " + type for type in types])
#pool.map(os.system, ["python3 analyseBulk.py " + type for type in types])
pool.map(os.system, ["python3 boxplotBulk.py " + type for type in types])

end = time.time()
print("Time elapsed: " + str(end - start))