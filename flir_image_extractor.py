#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import csv
import io
import json
import os
import os.path
import re
import subprocess
from math import exp, log, sqrt

import matplotlib.colors as mcolors
import mplcursors
import numpy as np
import scipy
import scienceplots
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import zoom

from colormaps import get_colormap_config, get_colormap_names

ref_mode = False
delta_mode = False
ref = 0


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class LogitNorm(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, weight=1):
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
    def __init__(
        self,
        vmin=None,
        vmax=None,
        clip=False,
        weight_low=1,
        weight_high=1,
        midpoint=None,
        transition_width=0.1,
    ):
        mcolors.Normalize.__init__(self, vmin, vmax, clip)
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.midpoint = (vmax + vmin) / 2.0 if midpoint is None else midpoint
        self.transition_width = transition_width * (vmax - vmin)

    def __call__(self, value, clip=None):
        x = np.clip((value - self.vmin) / (self.vmax - self.vmin), 1e-10, 1 - 1e-10)
        distance = np.abs(value - self.midpoint)
        scale = np.where(
            distance <= self.transition_width / 2,
            np.interp(distance, [0, self.transition_width / 2], [self.weight_low, self.weight_high]),
            np.where(value < self.midpoint, self.weight_low, self.weight_high),
        )
        return np.log10(1 + (x ** scale) * 9) / np.log10(10)


def sharpen_image(image, alpha=1.5):
    blurred = scipy.ndimage.gaussian_filter(image, sigma=1)
    unsharp_mask = image - blurred
    sharpened = image + alpha * unsharp_mask
    return np.clip(sharpened, np.min(image), np.max(image))


class FlirImageExtractor:
    def __init__(self, exiftool_path="exiftool", is_debug=False, plot_colormap="thermal"):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.plot_colormap = plot_colormap
        self.flir_img_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.default_distance = 1.0

        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None
        self.meta_data = None

    def process_image(self, flir_img_filename):
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        if self.get_image_type().upper().strip() == "TIFF":
            self.use_thumbnail = True
            self.fix_endian = False

        if not self.is_valid_parameter("EmbeddedImage"):
            self.use_thumbnail = True

        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np, self.meta_data = self.extract_thermal_image()

    def get_image_type(self):
        meta_json = subprocess.check_output(
            [self.exiftool_path, "-RawThermalImageType", "-j", self.flir_img_filename]
        )
        meta = json.loads(meta_json.decode())[0]
        return meta["RawThermalImageType"]

    def is_valid_parameter(self, param_name):
        meta = subprocess.check_output([self.exiftool_path, "-" + param_name, self.flir_img_filename]).decode()
        return len(meta) > 0

    def get_rgb_np(self):
        return self.rgb_image_np

    def get_thermal_np(self):
        return self.thermal_image_np

    def get_meta_data(self):
        return self.meta_data

    def extract_embedded_image(self):
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        return np.array(visual_img)

    def extract_thermal_image(self):
        meta_json = subprocess.check_output(
            [
                self.exiftool_path,
                self.flir_img_filename,
                "-Emissivity",
                "-SubjectDistance",
                "-AtmosphericTemperature",
                "-ReflectedApparentTemperature",
                "-IRWindowTemperature",
                "-IRWindowTransmission",
                "-RelativeHumidity",
                "-PlanckR1",
                "-PlanckB",
                "-PlanckF",
                "-PlanckO",
                "-PlanckR2",
                "-CameraModel",
                "-CameraSerialNumber",
                "-CameraPartNumber",
                "-j",
            ]
        )
        meta = json.loads(meta_json.decode())[0]

        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        subject_distance = self.default_distance
        if "SubjectDistance" in meta:
            subject_distance = FlirImageExtractor.extract_float(meta["SubjectDistance"])

        if self.fix_endian:
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00FF) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(
            lambda x: FlirImageExtractor.raw2temp(
                x,
                E=meta["Emissivity"],
                OD=subject_distance,
                RTemp=FlirImageExtractor.extract_float(meta["ReflectedApparentTemperature"]),
                ATemp=FlirImageExtractor.extract_float(meta["AtmosphericTemperature"]),
                IRWTemp=FlirImageExtractor.extract_float(meta["IRWindowTemperature"]),
                IRT=meta["IRWindowTransmission"],
                RH=FlirImageExtractor.extract_float(meta["RelativeHumidity"]),
                PR1=meta["PlanckR1"],
                PB=meta["PlanckB"],
                PF=meta["PlanckF"],
                PO=meta["PlanckO"],
                PR2=meta["PlanckR2"],
            )
        )
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np, meta

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340, PR2=0.012545258):
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        emiss_wind = 1 - IRT
        refl_wind = 0

        h2o = (RH / 100) * exp(1.5587 + 0.06939 * ATemp - 0.00027816 * ATemp ** 2 + 0.00000068455 * ATemp ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(-sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(-sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

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
        raw_obj = raw / E / tau1 / IRT / tau2 - raw_atm1_attn - raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn

        return PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15

    @staticmethod
    def extract_float(dirtystr):
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def plot(self):
        thermal_np = self.get_thermal_np()
        plot_data = self._build_plot_data(thermal_np)
        config = get_colormap_config(self.plot_colormap)
        plot_values = plot_data[config["image_source"]]
        norm = self._build_norm(config.get("norm"), thermal_np)
        imshow_kwargs = dict(config.get("imshow_kwargs", {}))

        if norm is None:
            imshow_kwargs.setdefault("vmin", np.min(thermal_np))
            imshow_kwargs.setdefault("vmax", np.max(thermal_np))

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.set_tight_layout(True)
        plt.style.use(["science", "no-latex"])

        im = ax.imshow(plot_values, cmap=config["cmap"], norm=norm, **imshow_kwargs)
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.062, label="Temperature [C]")

        meta_data = self.get_meta_data()
        print(meta_data.get("CameraModel"))
        print(meta_data.get("CameraSerialNumber"))

        def toggle_reference(event):
            global ref_mode
            global delta_mode
            if event.key == "r":
                ref_mode = True
                delta_mode = False
                print("Reference point mode activated.")
            elif event.key == "d":
                ref_mode = False
                delta_mode = True
                print("Delta calculation mode activated.")
            elif event.key == "n":
                ref_mode = False
                delta_mode = False
                print("Normal calculation mode activated.")

        crs1 = mplcursors.cursor(hover=0, multiple=True)
        crs2 = mplcursors.cursor(hover=2)
        ds = r"$\degree$"
        delta_symbol = r" $\Delta$"

        @crs1.connect("add")
        def on_add_primary(sel):
            global ref
            color = "#c2ccec" if ref_mode else "#f5f5dc"
            i, j = sel.index
            if delta_mode:
                sel.annotation.set_text(f"{delta_symbol}{np.round(plot_values[i, j] - ref, 2)}{ds}C")
            elif ref_mode:
                sel.annotation.set_text(f"{np.round(plot_values[i, j], 2)}{ds}C")
                ref = plot_values[i, j]
            else:
                sel.annotation.set_text(f"{np.round(plot_values[i, j], 2)} {ds}C")
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=0.5)
            sel.annotation.get_bbox_patch().set(fc=color, alpha=0.9)

        @crs2.connect("add")
        def on_add_secondary(sel):
            color = "#c2ccec" if ref_mode else "#f5f5dc"
            i, j = sel.index
            if delta_mode:
                sel.annotation.set_text(f"{delta_symbol}{np.round(plot_values[i, j] - ref, 2)}{ds}C")
            else:
                sel.annotation.set_text(f"{np.round(plot_values[i, j], 2)} {ds}C")
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=0.5)
            sel.annotation.get_bbox_patch().set(fc=color, alpha=0.9)

        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        fig.canvas.mpl_connect("key_press_event", toggle_reference)
        plt.show()

    def _build_plot_data(self, thermal_np, zoom_factor=6, sharpen_alpha=5):
        interpolated = np.array(zoom(thermal_np, zoom_factor, order=5))
        return {
            "raw": thermal_np,
            "interpolated": interpolated,
            "sharpened": sharpen_image(interpolated, sharpen_alpha),
        }

    def _build_norm(self, norm_config, thermal_np):
        if not norm_config:
            return None

        vmin = np.min(thermal_np)
        vmax = np.max(thermal_np)
        midpoint = norm_config.get("midpoint")
        if midpoint == "median":
            midpoint = np.median(thermal_np)
        elif midpoint == "center":
            midpoint = (vmax + vmin) / 2.0

        norm_type = norm_config["type"]
        if norm_type == "logit":
            return LogitNorm(vmin=vmin, vmax=vmax, weight=norm_config.get("weight", 1))
        if norm_type == "smooth_weighted_log":
            return SmoothWeightedLogNorm(
                vmin=vmin,
                vmax=vmax,
                weight_low=norm_config.get("weight_low", 1),
                weight_high=norm_config.get("weight_high", 1),
                midpoint=midpoint,
                transition_width=norm_config.get("transition_width", 0.1),
            )
        if norm_type == "midpoint":
            return MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)

        raise ValueError("Unsupported normalization type: {}".format(norm_type))

    def _render_thermal_image(self, thermal_np):
        plot_data = self._build_plot_data(thermal_np)
        config = get_colormap_config(self.plot_colormap)
        plot_values = plot_data[config["image_source"]]
        norm = self._build_norm(config.get("norm"), thermal_np)

        if norm is None:
            min_temp = np.min(thermal_np)
            max_temp = np.max(thermal_np)
            normalized = (plot_values - min_temp) / (max_temp - min_temp)
        else:
            normalized = norm(plot_values)

        cmap = cm.get_cmap(config["cmap"]) if isinstance(config["cmap"], str) else config["cmap"]
        return np.uint8(cmap(normalized) * 255)

    def save_images(self):
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()

        img_visual = Image.fromarray(rgb_np)
        img_thermal = Image.fromarray(self._render_thermal_image(thermal_np))

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
        with open(csv_filename, "w") as fh:
            writer = csv.writer(fh, delimiter=",")
            writer.writerow(["x", "y", "temp (c)"])

            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])

            writer.writerows(pixel_values)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract and visualize Flir Image data")
    parser.add_argument("-i", "--input", type=str, help="Input image. Ex. img.jpg", required=True)
    parser.add_argument("-p", "--plot", help="Generate a plot using matplotlib", required=False, action="store_true")
    parser.add_argument(
        "-c",
        "--colormap",
        type=str,
        choices=get_colormap_names(),
        default="thermal",
        help="Colormap to use for plotting. Default: %(default)s",
    )
    parser.add_argument("-exif", "--exiftool", type=str, help="Custom path to exiftool", required=False, default="exiftool")
    parser.add_argument("-csv", "--extractcsv", help="Export the thermal data per pixel encoded as csv file", required=False)
    parser.add_argument("-d", "--debug", help="Set the debug flag", required=False, action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    fie = FlirImageExtractor(
        exiftool_path=args.exiftool,
        is_debug=args.debug,
        plot_colormap=args.colormap,
    )
    fie.process_image(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()
