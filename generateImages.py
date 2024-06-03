import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
import os


#mi.set_variant('scalar_rgb')

class ImageGenerator:
    def __init__(self, scenePath, savePath, numLights):
        self.variant = mi.set_variant('llvm_ad_rgb')
        self.scene = mi.load_file(scenePath)
        self.params = mi.traverse(self.scene)
        self.savePath = savePath
        
        self.center_x = 0
        self.center_y = 0
        self.height = -0.7
        #self.radius = 0.25
        self.radius = 0.5 #narrow
        #self.radius = 1.4 #wide
        self.num_points = numLights

    def setSampleCircleParams(self, center_x, center_y, height, radius, num_points):
        self.center_x = center_x
        self.center_y = center_y
        self.height = height
        self.radius = radius
        self.num_points = num_points


    def sample_points_on_circle(self):
        # Calculate the angle between each sampled point
        angle_step = 2 * np.pi / self.num_points
        
        # Initialize an empty list to store the sampled points
        points = []
        
        # Sample points on the circle using the parametric equation
        for i in range(self.num_points):
            angle = i * angle_step #+ angle_step/2
            x = self.center_x + self.radius * np.cos(angle)
            y = self.center_y + self.radius * np.sin(angle)
            points.append((x, y,self.height))
        
        return points
    
    def setSavePath(self, savePath):
        self.savePath = savePath


    def setScene(self, scene):
        self.scene = mi.load_file(scene)
        self.params = mi.traverse(scene)
    
    def generate_images(self):
    # Load the scene
        circle_points = self.sample_points_on_circle()
        counter = 0

        #create lightPositions.npy file

        lightPositions = np.array(circle_points)

        #make homogenous transform matrices with the light positions

        transforms = []
        R = np.array([[1,0,0],[0,1,0],[0,0,1]])

        for i in range(lightPositions.shape[0]):

            T = np.array([[1,0,0,lightPositions[i,0]],[0,1,0,lightPositions[i,1]],[0,0,1,lightPositions[i,2]],[0,0,0,1]])
            transforms.append(T)
            
            




        #normalize the light positions

        #lightPositions = lightPositions / np.linalg.norm(lightPositions, axis=1)[:,None]
        
        np.save(self.savePath + "lightPositions",lightPositions)
        np.save(self.savePath + "TCP2CamPoses",transforms)

        # generate savepath for images
        if not os.path.exists(self.savePath + "images/"):
            os.makedirs(self.savePath + "images/")

       
        for point in circle_points:
            self.params['light2.position'] = point
            self.params.update()
            
            image = mi.render(self.scene)
            rgb = image[:,:,0:3]

            mi.util.write_bitmap(self.savePath + "images/" +"rgb" + str(counter).zfill(2) + ".png", rgb)
            mi.util.write_bitmap(self.savePath + "images/" +"rgb" + str(counter).zfill(2) + ".exr", rgb)
            npy_rgb = np.array(rgb)
            np.save(self.savePath + "images/" +"rgb" + str(counter).zfill(2)+".npy",npy_rgb)

            


            if counter == 0:
                trueNormals = image[:,:,4:7]

                #normals = (normals+1)/2 
                #plt.imshow(rgb ** (1.0 / 2.2))
                
                trueDepth = image[:,:,3]

                mi.util.write_bitmap(self.savePath + "images/" +"depth" + ".png", trueDepth)
                npy_normals = np.array(trueNormals)
                npy_depth = np.array(trueDepth)

                np.save(self.savePath  +"trueNormals",npy_normals)
                np.save(self.savePath +"trueDepth",npy_depth)            
                
            counter += 1


    









# image = mi.render(scene)

# rgb = image[:,:,0:3]

# depth = image[:,:,3]

# normals = image[:,:,4:7]

# normals = (normals+1)/2 


# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].imshow(rgb)
# ax[1].imshow(depth)
# ax[2].imshow(normals)


#plt.axis("off")
#plt.imshow(rgb ** (1.0 / 2.2))
#plt.imshow(normals)
#mi.util.write_bitmap("my_first_render.png", rgb)
#plt.show()
#mi.Bitmap(img).write('cbox.exr')