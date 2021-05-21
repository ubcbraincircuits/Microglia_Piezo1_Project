import colorcet as cc
import functions as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.measure as measure
import os
import tifffile

from cellpose import models
from skimage import measure
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

"""
Cellpose: a generalist algorithm for cellular segmentation
Carsen Stringer, Tim Wang, Michalis Michaelos, Marius Pachitariu
bioRxiv 2020.02.02.931238; doi: https://doi.org/10.1101/2020.02.02.931238
"""



class microglia():
    def __init__(self, path, hz, filterCells=False):
        self.HZ = hz
        # Open the image and return the two channels
        def openimages(self, path):
          raw_image = tifffile.imread(path)
          if len(raw_image.shape) ==4:
              return raw_image[ :, 0, :, :], raw_image[ :, 1, :, :]
          else:
              return raw_image[0, :, 0, :, :], raw_image[0, :, 1, :, :]

        self.cell_id = os.path.basename(path)[:-4]

        self.fl_image, self.bf_image = openimages(self, path)
        self.filterCells = filterCells

        # Generates masks of the dye filled cells
        def return_masks(self, raw_image, fcc=False):
            image = raw_image.copy()
            max_pro = np.max(image, axis=0)
            model = models.Cellpose(gpu=False, model_type='cyto')
            masks, _, _, _ = model.eval(max_pro, diameter=45, channels=[0,0])
            if fcc == True:
                masks = mpf.filter_cropped_cells(masks)
            return masks

        self.masks = return_masks(self, self.fl_image, fcc = self.filterCells)

        # Returns the raw intensity change of a ROI
        def rawIntensity(video_mask):
            video_mask_nan = video_mask.copy()
            video_mask_nan[video_mask_nan==0] = np.nan
            mean = np.nanmean(video_mask, axis=(1,2))
            return mean


        def roi2trace(self, raw_image, masks):
            trace_array = np.zeros((np.unique(masks).shape[0], raw_image.shape[0]))
            for i in list(np.unique(masks)):
                if i > 0:
                    temp_mask = masks.copy()
                    temp_mask[temp_mask != i] = 0
                    temp_mask[temp_mask == i] = 1
                    masked_ca = temp_mask * raw_image.copy()
                    masked_ca = masked_ca.astype('float')
                    trace_array[i,:] = rawIntensity(masked_ca)

            return trace_array

        self.raw_traces = roi2trace(self, self.fl_image, self.masks)

        self.dff = mpf.deltaFOverF0(self.raw_traces, hz=self.HZ)

        def save_traces(cellID, trace, deltaF):
            if not os.path.exists('output'):
                os.mkdir('output')
                os.mkdir('output/raw_trace/')
                os.mkdir('output/dff/')
            pd.DataFrame(trace).to_csv("output/raw_trace/%s_raw_trace.csv" %(cellID))
            pd.DataFrame(deltaF).to_csv("output/dff/%s_dff.csv" %(cellID))
        save_traces(self.cell_id, self.raw_traces, self.dff)

    def inspect_results(self):
        masks=self.masks.copy()
        trace_array=self.dff.copy()
        channel1 = self.fl_image.copy()
        channel2 =self.bf_image.copy()
        closed = masks.copy()
        contours = []
        mask_values = np.unique(masks)
        for i in range(len(mask_values)):
            closed = masks.copy()
            closed[closed !=mask_values[i]]=0
            contours.append(measure.find_contours(closed, .4))


        num_roi = np.unique(masks).shape[0]
        color_step =  (1/(num_roi+1))
        step_color = color_step
        color_list = []
        for i in range(num_roi):
            plot_color = cc.glasbey[i]
            color_list.append((plot_color))
            step_color += color_step


        def plot(Frame, ROI):
            contour = contours[ROI]

            plt.rcParams["figure.figsize"] = [16, 10]
            closed = masks.copy()


            fig, ax = plt.subplots(2)
            ax[0].imshow(channel2[Frame,:,:], cmap=plt.cm.gray)
            ax[0].imshow(channel1[Frame,:,:], cmap='magma', alpha=.5)

            ax[0].plot(contour[0][:, 1],contour[0][:, 0], linewidth=4, color=color_list[ROI])
            ax[1].plot(trace_array[ROI, :], color=color_list[ROI])
            ax[1].axvline(x=Frame, color=color_list[ROI], ls='--')
            plt.show()

        display(interact(plot, Frame=widgets.IntSlider(min=0, max=(channel1.shape[0]-1), step=1, value=0), ROI=widgets.IntSlider(min=1, max=(len(contours)-1), step=1, value=0),continuous_update=False))
