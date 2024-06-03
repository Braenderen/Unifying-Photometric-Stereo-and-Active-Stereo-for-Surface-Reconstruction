import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import cv2
import os, sys
import time


# ci_build_and_not_headless = False

# try:
#     from cv2.version import ci_build, headless
#     ci_and_not_headless = ci_build and not headless
# except:
#     pass
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_FONTDIR")

class ImageSegmenter:
    def __init__(self,savePath):
        self.model_type = "vit_h"
        self.model = sam_model_registry[self.model_type](checkpoint=f"./sam_{self.model_type}_4b8939.pth")
        self.model.to(device="cpu")
        self.predictor = SamPredictor(self.model)
        self.input_point = []
        self.input_label = []
        self.temp_mask = []
        self.mask = []
        self.maskReady = False
        self.savePath = savePath

    def onclick(self,event):
        print('in on click callback')
        if event.xdata != None and event.ydata != None:
            print(event.xdata, event.ydata)
            
            self.input_point = np.array([[event.xdata, event.ydata]])
            self.input_label = np.array([1])
            plt.close()

    def onclick2(self,event):

        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        
        color = np.array([255, 255, 255, 1])
        h, w = self.temp_mask.shape[-2:]
        #print("h",h)
        #print('temp_mask: ', self.temp_mask.shape)
        mask_image = self.temp_mask.reshape(h, w, 1)*color.reshape(1, 1, -1)
        #mask_image = temp_mask.reshape(h, w, 1)

        mask_rgb = mask_image[:, :, :1]

          

        self.mask = mask_rgb
        
        if event.button == 3:
            print('Saving mask to file')
            #cv2.imwrite(self.savePath + 'mask.png', self.mask)
            plt.close()
            self.maskReady = True
            return
        

        plt.close()

    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self,coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
            
    def show_box(self,box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


    def segment_image(self):

        #image = cv2.imread('./sensorPaper/images/pg/avg_image1.png')
        image = cv2.imread(self.savePath +"images/" +'depth.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        #masks, scores, logits = self.predictor.predict(multimask_output=True)

        #ax = plt.gca()
        fig = plt.gcf()
        plt.imshow(image)
        plt.axis('off')
        plt.title("Click on object of interest", fontsize=18)
        

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()
        

        masks, scores, logits = self.predictor.predict(
        point_coords=self.input_point,
        point_labels=self.input_label,
        multimask_output=True,
        )


        #input_point, input_label = [], []
       
     

        for i, (mask, score) in enumerate(zip(masks, scores)):
            if self.maskReady:
                break

            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca())
            self.show_points(self.input_point, self.input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            fig = plt.gcf()
            #global temp_mask
            self.temp_mask = mask

            cid = fig.canvas.mpl_connect('button_press_event', self.onclick2)
            plt.show() 


            fig.canvas.mpl_disconnect(cid)

        #plt.show()

        return self.mask
     





        









#fig.canvas.mpl_disconnect(cid)

# st = time.time()
# predictor.set_image(image)
# et = time.time()
# print('set image: ', et-st)




        




  
  