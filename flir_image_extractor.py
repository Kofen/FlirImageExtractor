#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import cmocean
import mplcursors
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, zoom
import scipy
import scienceplots

ref_mode = False
delta_mode = False
ref = 0

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0,0.5,1]
        return np.ma.masked_array(np.interp(value, x, y))


class LogitNorm(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, weight=1):
        """
        :param vmin: Minimum value of the input data
        :param vmax: Maximum value of the input data
        :param clip: If True values falling outside the range [vmin,vmax], are mapped to 0 or 1, whichever is closer
        :param weight: Weighting factor to adjust steepness at the ends
        """
        mcolors.Normalize.__init__(self, vmin, vmax, clip)
        self.weight = weight

    def __call__(self, value, clip=None):
        x = np.clip((value - self.vmin) / (self.vmax - self.vmin), 1e-10, 1 - 1e-10)
        x = x ** self.weight
        res = np.log10(x / (1 - x))
        return np.ma.masked_invalid(res)

    def inverse(self, value):
        y = 1 / (1 + np.exp(-value))
        y = y ** (1 / self.weight)
        return y * (self.vmax - self.vmin) + self.vmin

class SmoothWeightedLogNorm(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, weight_low=1, weight_high=1, midpoint=None, transition_width=0.1):
        """
        :param vmin: Minimum value of the input data
        :param vmax: Maximum value of the input data
        :param clip: If True, values falling outside the range [vmin,vmax] are mapped to 0 or 1, whichever is closer
        :param weight_low: Weighting factor for values below the midpoint
        :param weight_high: Weighting factor for values above the midpoint
        :param midpoint: The value that splits the data range into two. If None, it is set to the average of vmin and vmax
        :param transition_width: Width of the transition region around the midpoint as a fraction of the total range
        """
        mcolors.Normalize.__init__(self, vmin, vmax, clip)
        self.weight_low = weight_low
        self.weight_high = weight_high
        if midpoint is None:
            self.midpoint = (vmax + vmin) / 2.0
        else:
            self.midpoint = midpoint
        self.transition_width = transition_width * (vmax - vmin)

    def __call__(self, value, clip=None):
        # Normalize value to [0, 1]
        x = np.clip((value - self.vmin) / (self.vmax - self.vmin), 1e-10, 1 - 1e-10)
        # Calculate distance to the midpoint
        distance = np.abs(value - self.midpoint)
        # Calculate scaling factor based on distance to the midpoint
        scale = np.where(distance <= self.transition_width / 2,
                         np.interp(distance, [0, self.transition_width / 2], [self.weight_low, self.weight_high]),
                         np.where(value < self.midpoint, self.weight_low, self.weight_high))
        # Apply weighted log10 scaling
        x = np.log10(1 + (x ** scale) * 9) / np.log10(10)
        return x

class HammingNormalization(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        mcolors.Normalize.__init__(self, vmin, vmax, clip)
        self.hamming_window = np.hamming(1000)  # More points for a smoother transition
    
    def __call__(self, value, clip=None):
        # Normalize value to 0-1 range
        rescaled_value = (value - self.vmin) / (self.vmax - self.vmin)
        # Scale normalized value to hamming window size and apply window for emphasis
        index = np.clip((rescaled_value * (len(self.hamming_window) - 1)).astype(int), 0, len(self.hamming_window) - 1)
        return self.hamming_window[index]


def load_custom_colormap(file_path):
    # Load the RGB values
    rgb_values = np.load(file_path)
    
    # Create and return the colormap
    return ListedColormap(rgb_values)


def sharpen_image(image, alpha=1.5):
    # Apply a Gaussian blur to the image
    blurred = scipy.ndimage.gaussian_filter(image, sigma=1)

    # Calculate the difference (the "unsharp mask")
    unsharp_mask = image - blurred

    # Add the scaled unsharp mask back into the original image
    sharpened = image + alpha * unsharp_mask
    return np.clip(sharpened, np.min(image), np.max(image))  # Clip to maintain the original range

class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None

    pass

    def process_image(self, flir_img_filename):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        :param flir_img_filename:
        :return:
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        if not self.is_valid_parameter('EmbeddedImage'):
            self.use_thumbnail = True

        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np, self.meta_data = self.extract_thermal_image()

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]

        return meta['RawThermalImageType']

    def is_valid_parameter(self, param_name):
        """
        Check if the image has the requested parameter
        :return:
        """
        meta = subprocess.check_output(
            [self.exiftool_path, '-' + param_name, self.flir_img_filename]).decode()

        return True if len(meta) > 0 else False

    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np

    def get_meta_data(self):
        """
        Return the last meta data
        :return:
        """
        return self.meta_data
    
    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-CameraModel', '-CameraSerialNumber', '-CameraPartNumber', '-j'])
        meta = json.loads(meta_json.decode())[0]

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = self.default_distance
        if 'SubjectDistance' in meta:
            subject_distance = FlirImageExtractor.extract_float(meta['SubjectDistance'])

        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x, E=meta['Emissivity'], OD=subject_distance,
                                                                          RTemp=FlirImageExtractor.extract_float(
                                                                              meta['ReflectedApparentTemperature']),
                                                                          ATemp=FlirImageExtractor.extract_float(
                                                                              meta['AtmosphericTemperature']),
                                                                          IRWTemp=FlirImageExtractor.extract_float(
                                                                              meta['IRWindowTemperature']),
                                                                          IRT=meta['IRWindowTransmission'],
                                                                          RH=FlirImageExtractor.extract_float(
                                                                              meta['RelativeHumidity']),
                                                                          PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                          PF=meta['PlanckF'],
                                                                          PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np, meta

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                 PR2=0.012545258):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        # temperature from radiance
        temp_celcius = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        """
        Extract the float value of a string, helpful for parsing the exiftool data
        :return:
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def plot(self):
        """
        Plot the rgb + thermal image (easy to see the pixel values)
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()
        

        fig, ax = plt.subplots(1,1, figsize=(10,8))
        #im = [None]*len(ax)
        
        fig.set_tight_layout(True)
        plt.style.use(['science', 'no-latex'])
        # Assuming `data` is your 2D numpy array representing the thermal image
        # Interpolate to increase resolution, if needed
        zoom_factor = 6  # Specify the zoom factor here
        alpha = 0.3
        alpha = 5
        interpolated_data = zoom(thermal_np, zoom_factor, order=5)#, mode='mirror')  # order=3 for bicubic
        
        # Convert back to numpy array if needed
        thermal_int_np = np.array(interpolated_data)
        sharpened_data = sharpen_image(thermal_int_np, alpha)
        #print(np.median(thermal_np),np.median(new))
        meta_data = self.get_meta_data() 
        
        ### High constrast mode E75 ####
        """
        cmap = load_custom_colormap("flir_high_contrast.npy")
        
        # This also looks good, but is further from the Flir plot
        #norm = SmoothWeightedLogNorm(vmin=np.min(thermal_np), vmax=np.max(thermal_np), weight_low=1, weight_high=1.8, midpoint=np.median(thermal_np), transition_width=1)
        
        norm = SmoothWeightedLogNorm(vmin=np.min(thermal_np), vmax=np.max(thermal_np), weight_low=2, weight_high=1.8, midpoint=np.median(thermal_np), transition_width=1)
        #norm = MidpointNormalize(vmin=np.min(thermal_np), vmax=np.max(thermal_np), midpoint=(np.max(thermal_np)+np.min(thermal_np))/2)
        #norm = MidpointNormalize(vmin=np.min(thermal_np), vmax=np.max(thermal_np), midpoint=np.median(thermal_np))
        #norm = HammingNormalization(vmin=np.min(thermal_int_np), vmax=thermal_int_np.max())
        im = ax.imshow(thermal_int_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(sharpened_data, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """
        ###########################
        

        ### Normal Flir E75 ###
        # But PCB img taken with high contrast map is saturated AF --- Something is wrong in the mapping, fix it!
        """
        cmap = load_custom_colormap("flir.npy")
        im = ax.imshow(thermal_int_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(sharpened_data, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """

        ##########################

        
        ### ocean thermal Flir E75 ###
        """
        cmap = cmocean.cm.thermal
        im = ax.imshow(thermal_int_np, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax.imshow(sharpened_data, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """

        ##########################


        ### Seismic Flir E75 ###
        """
        cmap = 'seismic'
        norm = SmoothWeightedLogNorm(vmin=np.min(thermal_np), vmax=np.max(thermal_np), weight_low=2, weight_high=1.8, midpoint=np.median(thermal_np), transition_width=1)
        im = ax.imshow(thermal_int_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(sharpened_data, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """
        ##########################
        

        ##########################

        ### Hot Flir E75 ###
        """
        cmap = 'hot'
        norm = SmoothWeightedLogNorm(vmin=np.min(thermal_np), vmax=np.max(thermal_np), weight_low=2, weight_high=1.8, midpoint=np.median(thermal_np), transition_width=1)
        im = ax.imshow(thermal_int_np, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(thermal_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax.imshow(sharpened_data, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """
        ##########################

        ##########################
        ### Plasma Flir E75 ###
        """
        #cmap = None 
        cmap = 'plasma' 
        #cmap = load_custom_colormap("flir_high_contrast.npy")
        #ax[1].imshow(new, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        norm = SmoothWeightedLogNorm(vmin=np.min(thermal_np), vmax=np.max(thermal_np), weight_low=6, weight_high=0.5, midpoint=np.median(thermal_np), transition_width=10)
        im = ax.imshow(thermal_int_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[0].imshow(sharpened_data, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """
       
       ##########################

        ##########################
        ### Fluke Ti10 colormap Flir E75 ###
        """
        cmap = load_custom_colormap("fluke.npy")
        #cmap = load_custom_colormap("flir_high_contrast.npy")
        #im[1] = ax[1].imshow(thermal_int_np, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        im = ax.imshow(sharpened_data, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #norm = SmoothWeightedLogNorm(vmin=np.min(thermal_np), vmax=np.max(thermal_np), weight_low=1, weight_high=1.8, midpoint=np.median(thermal_np), transition_width=1)
        #im = ax.imshow(thermal_int_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[0].imshow(sharpened_data, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.29), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """
       
       ##########################
        
        ### Plain vanilla ###
        #"""
        #cmap = "inferno"
        #cmap = load_custom_colormap("flir.npy")
        #cmap = cmocean.cm.oxy
        cmap = cmocean.cm.thermal
        #cmap = cmocean.cm.topo
        im = ax.imshow(thermal_int_np, cmap=cmap, interpolation='nearest', aspect='auto', vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #"""
        ######################
       
       ##########################
        
        ### Oxygen ###
        """
        #cmap = "inferno"
        #cmap = load_custom_colormap("flir.npy")
        cmap = cmocean.cm.oxy
        #cmap = cmocean.cm.thermal
        #cmap = cmocean.cm.topo
        im = ax.imshow(thermal_int_np, cmap=cmap, interpolation='nearest', aspect='auto')#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #norm = MidpointNormalize(vmin=np.min(thermal_np), vmax=np.max(thermal_np), midpoint=(np.max(thermal_np)+np.min(thermal_np))/2)
        #norm = MidpointNormalize(vmin=np.min(thermal_np), vmax=np.max(thermal_np), midpoint=np.median(thermal_np))
        #im = ax.imshow(thermal_int_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        """
        ######################

        # Ocean thermal with log, will be lit af, add
        # Oxy gives a nice contrasst, usefull, add


        #print(len(ax))
        #"for i in range(len(ax)):
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.062, label="Temperature [C]")
       
        print(meta_data.get('CameraModel'))
        print(meta_data.get('CameraSerialNumber'))
        
        def toggle_reference(event):
            global ref_mode
            global delta_mode
            if event.key == 'r':
                ref_mode = True
                delta_mode = False
                # Activate reference point setting mode
                print("Reference point mode activated.")
            elif event.key == 'd':
                ref_mode = False
                delta_mode = True
                # Activate delta calculation mode
                print("Delta calculation mode activated.")
            elif event.key == 'n':
                ref_mode = False
                delta_mode = False
                # Activate delta calculation mode
                print("Normal calculation mode activated.")



        crs1 = mplcursors.cursor(hover=0, multiple=True)
        crs2 = mplcursors.cursor(hover=2)

        #crs1.connect('add', lambda sel: sel.annotation.set_text('{}C'.format(sel.target[0,1])))
        #crs1.connect('add', lambda sel: sel.annotation.get_bbox_patch().set(fc='#f5f5dc', alpha=0.9))
        #crs1.connect('add', lambda sel: sel.annotation.get_bbox_patch().set(fc='#f5f5dc', alpha=0.9))
        ds = r'$\degree$' 
        Delta = r' $\Delta$'
        @crs1.connect("add")
        def on_add(sel):
            global ref
            color = '#c2ccec' if ref_mode else '#f5f5dc'
            i,j = sel.index
            if delta_mode:
                sel.annotation.set_text(f"{Delta}{np.round(thermal_int_np[i,j]-ref,2)}{ds}C")
            elif ref_mode:
                sel.annotation.set_text(f"{np.round(thermal_int_np[i,j],2)}{ds}C")
                ref = thermal_int_np[i,j]
            else:
                sel.annotation.set_text(f"{np.round(thermal_int_np[i,j],2)} {ds}C")
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)
            sel.annotation.get_bbox_patch().set(fc=color, alpha=0.9)
        
        @crs2.connect("add") 
        def on_add(sel):
            color = '#c2ccec' if ref_mode else '#f5f5dc'
            i,j = sel.index
            if delta_mode:
                sel.annotation.set_text(f"{Delta}{np.round(thermal_int_np[i,j]-ref,2)}{ds}C")
            else:
                sel.annotation.set_text(f"{np.round(thermal_int_np[i,j],2)} {ds}C")
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)
            sel.annotation.get_bbox_patch().set(fc=color, alpha=0.9)


        plt.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)
        fig.canvas.mpl_connect('key_press_event', toggle_reference)
        plt.show()
       
    def save_images(self):
        """
        Save the extracted images
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np, _ = self.extract_thermal_image()

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        image_filename = fn_prefix + self.image_suffix
        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))

        img_visual.save(image_filename)
        img_thermal.save(thermal_filename)

    def export_thermal_to_csv(self, csv_filename):
        """
        Convert thermal data in numpy to json
        :return:
        """

        with open(csv_filename, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'temp (c)'])

            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])

            writer.writerows(pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=True)
    parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
                        required=False)
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)
    fie.process_image(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()




#### CUM DUMP ####



        #im = ax[1].imshow(thermal_np, cmap=cmap, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax[0].imshow(thermal_np, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax[1].imshow(thermal_np, norm=colors.LogNorm(vmin=thermal_np.min(), vmax=thermal_np.max()), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(thermal_np, norm=colors.TwoSlopeNorm(vcenter=np.median(thermal_np), vmin=thermal_np.min(), vmax=thermal_np.max()), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(thermal_np, norm=colors.TwoSlopeNorm(vcenter=np.median(thermal_np), vmin=thermal_np.min(), vmax=thermal_np.max()), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax[1].imshow(thermal_np, norm=colors.AsinhNorm(linear_width=1), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax[0].imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.57), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        
        #ax[0].imshow(thermal_np, norm=norm, cmap=cmap)#LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.25), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[0].imshow(thermal_np, norm=colors.PowerNorm(gamma=2), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(rgb_np, cmap=cmap)
        #ax[1].imshow(new, norm=colors.PowerNorm(gamma=0.95), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(new, vmin=np.min(new)+1,vmax=np.max(new)-2, cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #ax[1].imshow(new, vmin=np.min(new),vmax=np.max(new), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax[1].imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.9), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        #im = ax[1].imshow(thermal_np, norm=LogitNorm(vmin=np.min(thermal_np),vmax=np.max(thermal_np), weight=0.9), cmap=cmap)#, vmin=np.min(thermal_np), vmax=np.max(thermal_np))
        


