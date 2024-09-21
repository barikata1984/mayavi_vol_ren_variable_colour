import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from mayavi import mlab
from mayavi.core.lut_manager import LUTManager
from moviepy import editor
from tvtk.util import ctf  # ColorTransferFunction

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm


def _get_custom_colormap(name, num_colors=256, opacity=1., transparent_input=-1.):
    # Get a mayavi colormap as a numpy array
    lm = LUTManager(number_of_colors=num_colors, lut_mode=name, show_scalar_bar=True)
    rgbs = lm.lut.table.to_array()[:, :3]  # ~ (:, RGBA) -> (:, RGB)
    rgbs = rgbs.astype(float) / (num_colors - 1)  # [0.0, 1.0]**3
    # Setup a color transfer function with the above-generated colormap
    _ctf = ctf.ColorTransferFunction()
    for i, rgb in enumerate(rgbs):
        _ctf.add_rgb_point(i/(num_colors - 1), *rgb)

    # Setup a opacity transfer function
    _otf = ctf.PiecewiseFunction()
    _otf.add_point(transparent_input, 0.)  # transparent
    _otf.add_point(0., opacity)
    _otf.add_point(1, opacity)
    # Extend the input value range to allow transparent plot
    _ctf.range = [transparent_input, 1]

    return _ctf, _otf


def _init_mayavi_volume_rendering(s, x=None, y=None, z=None, _ctf=None, _otf=None, shade=True, size=720):
    # Initial rendering of mass distr
    black = (0, 0, 0)
    white = (1, 1, 1)
    figure = mlab.figure(bgcolor=white, fgcolor=black, size=(size, size))

    inputs = [s]
    if x is not None and y is not None and z is not None:
        inputs = [x, y, z] + inputs

    scalar_field = mlab.pipeline.scalar_field(*inputs)
    volume = mlab.pipeline.volume(scalar_field, figure=figure)

    if _ctf is not None:
        volume._ctf = _ctf
        volume._volume_property.set_color(_ctf)
        volume.update_ctf = True

    if _otf is not None:
        volume._otf = _otf
        volume._volume_property.set_scalar_opacity(_otf)

    volume._volume_property.shade = shade

    return figure, volume


def _generate_mass_distr_video(mass_distr_img_dir, output_name, fps):
    # Get a list of file names in the image folder
    input_ext = ".png"
    imgs = sorted(os.listdir(mass_distr_img_dir))
    images = [img for img in imgs if img.endswith(input_ext)]
    image_paths = [os.path.join(mass_distr_img_dir, img) for img in images]

    # Ganarate a video and save it on local
    codec = "libx264"
    video_name = os.path.join(mass_distr_img_dir, output_name)
    clip = editor.ImageSequenceClip(image_paths, fps=fps)
    clip.write_videofile(f"{video_name}.mp4", codec=codec, fps=fps)
    clip.write_gif(f"{video_name}.gif", fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir")
    parser.add_argument("--lod", type=int, default=7)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--matplotlib", action="store_true")
    parser.add_argument("--axes", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--generate-video", action="store_true")
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    print(f"{args=}\n")

    # Main part ========================================================
    mlab.options.offscreen = args.offscreen

    object_dir = Path(args.object_dir)
    csv_filepath = object_dir / "ground_truth.csv"
    metadata_df = pd.read_csv(csv_filepath,
                          nrows=1,  # num metadata rows after the metadata header
                          )


    print(f"{metadata_df}")

    main_df = pd.read_csv(csv_filepath,
                          skiprows=2,  # exclude the metadata and its header
                          ).loc[:, "x":"mass_density"]

    print(f"{main_df.to_numpy().shape=}")

    _x, _y, _z, _mass_distr = main_df.to_numpy().T[:4]

    res = math.ceil(math.pow(_x.shape[0], 1/3))
    ndc_coords = (2 * np.arange(0, res) - res + 1.) / res
    print(f"{res=}")
    aabb_scale = _x.max() / ndc_coords.max()
    print(f"{aabb_scale=}")  # [m]

    x = _x / aabb_scale
    y = _y / aabb_scale
    z = _z / aabb_scale

    # set plot data
    cmap = cm.get_cmap(args.cmap)
    c = _mass_distr / _mass_distr.max()
    s = np.zeros_like(_mass_distr)
    s[_mass_distr.nonzero()] = 1

    matplotlib_scatter = args.matplotlib
    if matplotlib_scatter:
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')
        size_const = 0.3
        ax.scatter(x, y, z, s=size_const*s, c=c, alpha=args.alpha, cmap=cmap)
        plt.show(block=False)

    # volume render with mayavi ========================================
    # Setup input data
    transparent_input = -1.
    meshgrid_shape = (res, res, res)
    scalars = transparent_input * np.ones(meshgrid_shape)
    zero_to_one_md = _mass_distr / _mass_distr.max()
    zero_to_one_md = zero_to_one_md.reshape(meshgrid_shape)
    scalars[zero_to_one_md.nonzero()] = zero_to_one_md[zero_to_one_md.nonzero()]

    print(f"Total mass: {_mass_distr[_mass_distr.nonzero()].sum()}")

    _ctf, _otf = _get_custom_colormap(args.cmap,
                                      num_colors=256,
                                      opacity=args.alpha,
                                      transparent_input=transparent_input)

    figure, volume = _init_mayavi_volume_rendering(scalars,
                                                   x=x.reshape(meshgrid_shape),
                                                   y=y.reshape(meshgrid_shape),
                                                   z=z.reshape(meshgrid_shape),
                                                   _ctf=_ctf,
                                                   _otf=_otf,
                                                   shade=False,
                                                   size=args.size
                                                   )

    mlab.text(0.02, 0.05,  # location
              f"object: {metadata_df.at[0, 'id']}",
              figure=figure,
              #line_width=.5,
              ).property.font_family = "courier"

    mlab.text(0.02, 0.01,  # location
              f"aabb scale: {metadata_df.at[0, 'aabb_scale']} [m]",
              figure=figure,
              #line_width=.5,
              ).property.font_family = "courier"

    if args.axes:
        mlab.axes(figure=figure,
                  x_axis_visibility = True,
                  xlabel = "X",
                  y_axis_visibility = True,
                  ylabel = "Y",
                  z_axis_visibility = True,
                  zlabel = "Z",
                  )

    if args.generate_video:
        max_epochs = 120
        num_mayavi_camera_turn = 2
        azimuth, _, _, _ = mlab.view()
        azimuth_tick = num_mayavi_camera_turn * 360.0 / max_epochs

        mass_distr_img_dir = object_dir / "mass_distr"
        os.makedirs(mass_distr_img_dir, exist_ok=True)

        for epoch in range(max_epochs):
            distance = 5.9  # this val provides the tightest extra space around the bbox
            azimuth = (azimuth + azimuth_tick) % 360.0
            mlab.view(azimuth=azimuth, distance=distance, figure=figure)

            filename = f"{epoch:04}.png"
            mlab.savefig(str(mass_distr_img_dir / filename), figure=figure)

        _generate_mass_distr_video(mass_distr_img_dir, "mass_distr", fps=args.fps)
    else:
        mlab.show()
