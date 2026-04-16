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
import numpy as np
import scipy
import scienceplots
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from PIL import Image
from scipy.ndimage import zoom

from colormaps import get_colormap_config, get_colormap_names


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
        config = get_colormap_config(self.plot_colormap)
        data_min = float(np.min(thermal_np))
        data_max = float(np.max(thermal_np))
        scale_step = max((data_max - data_min) / 200.0, 0.01)
        slider_state = {"updating": False}

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.style.use(["science", "no-latex"])
        plt.subplots_adjust(bottom=0.28)

        plot_state = {
            "plot_values": None,
            "scale_min": data_min,
            "scale_max": data_max,
            "sharpen_alpha": 5.0,
            "markers": [],
            "reference": None,
            "tool": "marker",
        }

        plot_values = self._get_plot_values(thermal_np, config, plot_state["sharpen_alpha"])
        plot_state["plot_values"] = plot_values
        norm = self._build_norm(config.get("norm"), thermal_np, plot_state["scale_min"], plot_state["scale_max"])
        imshow_kwargs = dict(config.get("imshow_kwargs", {}))
        if norm is None:
            imshow_kwargs["vmin"] = plot_state["scale_min"]
            imshow_kwargs["vmax"] = plot_state["scale_max"]
        im = ax.imshow(plot_values, cmap=config["cmap"], norm=norm, **imshow_kwargs)
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.062, label="Temperature [C]")

        meta_data = self.get_meta_data()
        print(meta_data.get("CameraModel"))
        print(meta_data.get("CameraSerialNumber"))

        marker_styles = {
            "marker": {
                "color": "#f5f5dc",
                "marker": "o",
                "label": "Marker",
                "offset": (14, 14),
            },
            "reference": {
                "color": "#c2ccec",
                "marker": "s",
                "label": "Reference",
                "offset": (14, -18),
            },
            "delta": {
                "color": "#f5b7b1",
                "marker": "^",
                "label": "Delta",
                "offset": (-14, 14),
            },
        }
        status_text = fig.text(0.02, 0.24, "", fontsize=10)

        def marker_value(marker):
            return float(plot_state["plot_values"][marker["y"], marker["x"]])

        def set_tool(tool_name):
            plot_state["tool"] = tool_name
            status_labels = {
                "marker": "Click image to add a marker",
                "reference": "Click image to set the delta reference marker",
                "delta": "Click image to add a delta marker",
                "delete": "Click an existing marker to delete it",
            }
            status_text.set_text("Tool: {}".format(status_labels[tool_name]))
            fig.canvas.draw_idle()

        def marker_text(marker):
            value = marker_value(marker)
            if marker["kind"] == "reference":
                return "Ref {:.2f} °C".format(value)
            if marker["kind"] == "delta":
                reference_marker = plot_state["reference"]
                if reference_marker is None:
                    return "Δ? °C\n{:.2f} °C".format(value)
                delta_value = value - marker_value(reference_marker)
                return "Δ{:.2f} °C\n{:.2f} °C".format(delta_value, value)
            return "{:.2f} °C".format(value)

        def refresh_marker(marker):
            style = marker_styles[marker["kind"]]
            marker["point"].set_offsets([[marker["x"], marker["y"]]])
            marker["annotation"].xy = (marker["x"], marker["y"])
            marker["annotation"].set_text(marker_text(marker))
            marker["annotation"].set_position(style["offset"])
            marker["annotation"].get_bbox_patch().set(fc=style["color"], alpha=0.9)

        def refresh_all_markers():
            for marker in plot_state["markers"]:
                refresh_marker(marker)
            fig.canvas.draw_idle()

        def remove_marker(marker):
            marker["point"].remove()
            marker["annotation"].remove()
            plot_state["markers"].remove(marker)
            if plot_state["reference"] is marker:
                plot_state["reference"] = None
            refresh_all_markers()

        def add_marker(x_idx, y_idx, kind):
            if kind == "reference" and plot_state["reference"] is not None:
                remove_marker(plot_state["reference"])

            style = marker_styles[kind]
            point = ax.scatter(
                [x_idx],
                [y_idx],
                s=90,
                c=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
            annotation = ax.annotate(
                "",
                xy=(x_idx, y_idx),
                xytext=style["offset"],
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc=style["color"], alpha=0.9),
                arrowprops=dict(arrowstyle="simple", fc="white", alpha=0.5),
                fontsize=9,
                zorder=5,
            )

            marker = {
                "kind": kind,
                "x": x_idx,
                "y": y_idx,
                "point": point,
                "annotation": annotation,
            }
            plot_state["markers"].append(marker)

            if kind == "reference":
                plot_state["reference"] = marker

            refresh_all_markers()

        def find_nearest_marker(x_value, y_value, max_distance=12):
            if not plot_state["markers"]:
                return None

            nearest_marker = None
            nearest_distance = None
            for marker in plot_state["markers"]:
                distance = np.hypot(marker["x"] - x_value, marker["y"] - y_value)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_marker = marker

            if nearest_distance is None or nearest_distance > max_distance:
                return None
            return nearest_marker

        def update_plot_image():
            plot_state["plot_values"] = self._get_plot_values(thermal_np, config, plot_state["sharpen_alpha"])
            im.set_data(plot_state["plot_values"])

            norm = self._build_norm(
                config.get("norm"),
                thermal_np,
                plot_state["scale_min"],
                plot_state["scale_max"],
            )
            if norm is None:
                im.set_norm(None)
                im.set_clim(plot_state["scale_min"], plot_state["scale_max"])
            else:
                im.set_norm(norm)
            refresh_all_markers()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            x_idx = int(np.clip(np.round(event.xdata), 0, plot_state["plot_values"].shape[1] - 1))
            y_idx = int(np.clip(np.round(event.ydata), 0, plot_state["plot_values"].shape[0] - 1))

            if plot_state["tool"] == "delete":
                marker = find_nearest_marker(x_idx, y_idx)
                if marker is not None:
                    remove_marker(marker)
                return

            if plot_state["tool"] == "marker":
                add_marker(x_idx, y_idx, "marker")
            elif plot_state["tool"] == "reference":
                add_marker(x_idx, y_idx, "reference")
            elif plot_state["tool"] == "delta":
                add_marker(x_idx, y_idx, "delta")

        def on_scale_change(_value):
            if slider_state["updating"]:
                return

            scale_min = min_slider.val
            scale_max = max_slider.val
            if scale_min >= scale_max:
                slider_state["updating"] = True
                if _value == scale_min:
                    min_slider.set_val(scale_max - scale_step)
                else:
                    max_slider.set_val(scale_min + scale_step)
                slider_state["updating"] = False
                scale_min = min_slider.val
                scale_max = max_slider.val

            plot_state["scale_min"] = scale_min
            plot_state["scale_max"] = scale_max
            update_plot_image()

        def on_sharpen_change(value):
            plot_state["sharpen_alpha"] = value
            update_plot_image()

        button_specs = [
            ("Add Marker", [0.12, 0.18, 0.16, 0.05], "marker"),
            ("Set Reference", [0.31, 0.18, 0.16, 0.05], "reference"),
            ("Add Delta", [0.50, 0.18, 0.16, 0.05], "delta"),
            ("Delete Marker", [0.69, 0.18, 0.16, 0.05], "delete"),
        ]

        widgets = []
        for label, rect, tool_name in button_specs:
            button = Button(fig.add_axes(rect), label)
            button.on_clicked(lambda _event, tool=tool_name: set_tool(tool))
            widgets.append(button)

        min_slider = Slider(
            fig.add_axes([0.12, 0.11, 0.73, 0.03]),
            "Scale Min",
            data_min,
            data_max - scale_step,
            valinit=data_min,
        )
        max_slider = Slider(
            fig.add_axes([0.12, 0.07, 0.73, 0.03]),
            "Scale Max",
            data_min + scale_step,
            data_max,
            valinit=data_max,
        )
        sharpen_slider = Slider(
            fig.add_axes([0.12, 0.03, 0.73, 0.03]),
            "Sharpen",
            0.0,
            10.0,
            valinit=plot_state["sharpen_alpha"],
        )
        widgets.extend([min_slider, max_slider, sharpen_slider])
        fig._flir_widgets = widgets

        min_slider.on_changed(on_scale_change)
        max_slider.on_changed(on_scale_change)
        sharpen_slider.on_changed(on_sharpen_change)

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
        fig.canvas.mpl_connect("button_press_event", on_click)
        set_tool("marker")
        plt.show()

    def _build_plot_data(self, thermal_np, zoom_factor=6, sharpen_alpha=5):
        interpolated = np.array(zoom(thermal_np, zoom_factor, order=5))
        return {
            "raw": thermal_np,
            "interpolated": interpolated,
            "sharpened": sharpen_image(interpolated, sharpen_alpha),
        }

    def _build_norm(self, norm_config, thermal_np, vmin=None, vmax=None):
        if not norm_config:
            return None

        if vmin is None:
            vmin = np.min(thermal_np)
        if vmax is None:
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

    def _get_plot_values(self, thermal_np, config, sharpen_alpha):
        plot_data = self._build_plot_data(thermal_np, sharpen_alpha=sharpen_alpha)
        return plot_data[config["image_source"]]

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
