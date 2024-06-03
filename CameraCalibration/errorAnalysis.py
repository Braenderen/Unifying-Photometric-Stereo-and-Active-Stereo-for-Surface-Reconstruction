import numpy as np
import matplotlib.pyplot as plt


path = './calibMay22/pg/'

errors = np.load(path + 'perViewErrors.npy')
print(errors.shape)
rms = np.sqrt(np.mean(errors**2))
print(rms)
#plt.figure(figsize=(4, 3))
plt.plot(errors)
plt.grid()
plt.ylabel('Reprojection Error (pixels)')
plt.xlabel('Image Index')
plt.savefig("reprojectionError.pdf", format="pdf", bbox_inches="tight")
plt.show()