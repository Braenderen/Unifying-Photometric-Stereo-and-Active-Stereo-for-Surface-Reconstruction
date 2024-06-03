import os
import time

configs = [
            # "./test/configs/difuse_3lights.yaml",
            # "./test/configs/difuse_4lights.yaml",
            # "./test/configs/difuse_5lights.yaml",
            # "./test/configs/difuse_6lights.yaml",
            # "./test/configs/difuse_7lights.yaml",
            # "./test/configs/difuse_8lights.yaml",
            # "./test/configs/difuse_9lights.yaml",
            # "./test/configs/difuse_10lights.yaml",
            # "./test/configs/difuse_15lights.yaml",
            # "./test/configs/difuseWide_3lights.yaml",
            # "./test/configs/difuseWide_4lights.yaml",
            # "./test/configs/difuseWide_5lights.yaml",
            # "./test/configs/difuseWide_6lights.yaml",
            # "./test/configs/difuseWide_7lights.yaml",
            # "./test/configs/difuseWide_8lights.yaml",
            # "./test/configs/difuseWide_9lights.yaml",
            # "./test/configs/difuseWide_10lights.yaml",
            # "./test/configs/difuseWide_15lights.yaml",
            # "./test/configs/plastic_3lights.yaml", 
            # "./test/configs/plastic_4lights.yaml", 
            # "./test/configs/plastic_5lights.yaml", 
            # "./test/configs/plastic_6lights.yaml", 
            # "./test/configs/plastic_7lights.yaml", 
            # "./test/configs/plastic_8lights.yaml", 
            # "./test/configs/plastic_9lights.yaml", 
            # "./test/configs/plastic_10lights.yaml", 
            "./test/configs/plastic_15lights.yaml",
            # "./test/configs/plasticNarrow_3lights.yaml",
            # "./test/configs/plasticNarrow_4lights.yaml",
            # "./test/configs/plasticNarrow_5lights.yaml",
            # "./test/configs/plasticNarrow_6lights.yaml",
            # "./test/configs/plasticNarrow_7lights.yaml",
            # "./test/configs/plasticNarrow_8lights.yaml",
            # "./test/configs/plasticNarrow_9lights.yaml",
            # "./test/configs/plasticNarrow_10lights.yaml",
            "./test/configs/plasticNarrow_15lights.yaml",
            # "./test/configs/Adifuse_3lights.yaml",
            # "./test/configs/Adifuse_4lights.yaml",
            # "./test/configs/Adifuse_5lights.yaml",
            # "./test/configs/Adifuse_6lights.yaml",
            # "./test/configs/Adifuse_7lights.yaml",
            # "./test/configs/Adifuse_8lights.yaml",
            # "./test/configs/Adifuse_9lights.yaml",
            # "./test/configs/Adifuse_10lights.yaml",
            # "./test/configs/Adifuse_15lights.yaml",
            # "./test/configs/Aplastic_3lights.yaml",
            # "./test/configs/Aplastic_4lights.yaml",
            # "./test/configs/Aplastic_5lights.yaml",
            # "./test/configs/Aplastic_6lights.yaml",
            # "./test/configs/Aplastic_7lights.yaml",
            # "./test/configs/Aplastic_8lights.yaml",
            # "./test/configs/Aplastic_9lights.yaml",
            # "./test/configs/Aplastic_10lights.yaml",
            # "./test/configs/Aplastic_15lights.yaml",
            # "./test/configs/metal_3lights.yaml",
            # "./test/configs/metal_4lights.yaml",
            # "./test/configs/metal_5lights.yaml",
            # "./test/configs/metal_6lights.yaml",
            # "./test/configs/metal_7lights.yaml",
            # "./test/configs/metal_8lights.yaml",
            # "./test/configs/metal_9lights.yaml",
            # "./test/configs/metal_10lights.yaml",
            # "./test/configs/metal_15lights.yaml",
            # "./test/configs/metalNarrow_3lights.yaml",
            # "./test/configs/metalNarrow_4lights.yaml",
            # "./test/configs/metalNarrow_5lights.yaml",
            # "./test/configs/metalNarrow_6lights.yaml",
            # "./test/configs/metalNarrow_7lights.yaml",
            # "./test/configs/metalNarrow_8lights.yaml",
            # "./test/configs/metalNarrow_9lights.yaml",
            # "./test/configs/metalNarrow_10lights.yaml",
            # "./test/configs/metalNarrow_15lights.yaml"
            ]

start = time.time()
for config in configs:
    # if os.path.exists(config):
    #     print("Running normals test for " + config)
    # else:
    #     print("Config file " + config + " does not exist")

    os.system("python3 normals_test.py " + config)
    print("")
    print("")

end = time.time()
print("Time elapsed: " + str(end - start))
    