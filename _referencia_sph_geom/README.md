# Geometry generation for SPH

This repository contains the code and manuscript for research done on
generating suitable input for simulation of flow past generic geometries with
SPH. Essentially this is a preprocessor.

## File tree

The following are the contents of the of this repository. The `code/` directory
contains algorithms and examples that are used in the manuscript, the
`manuscript/` directory contains the LaTeX code for the generation of the
manuscript, the `automate.py` file runs all the test cases that are used in the
present study, the file `requirements.txt` specifies the dependencies that are
required to run the current code and the license is at `LICENSE.txt`.

```bash
.
├── automate.py
├── code
│   ├── case_2d_annular_cylinder.py
│   ├── case_2d_bounded_airfoil.py
│   ├── case_2d_bounded_cylinder.py
│   ├── case_2d_bounded_object.py
│   ├── case_2d_periodic.py
│   ├── case_3d_bounded_lattice.py
│   ├── case_3d_bounded_sphere.py
│   ├── case_3d_bunny.py
│   ├── colagrossi.py
│   ├── conftest.py
│   ├── data
│   │   ├── bun_zipper_res2.stl
│   │   ├── ellipsoid.stl
│   │   ├── NACA0015.dat
│   │   ├── starfish.csv
│   │   └── starfish_sorted.csv
│   │   └── zig-zag.csv
│   ├── jiang.py
│   ├── particle_packing.py
│   ├── repulsion_force_plot.py
│   ├── sort_points.py
│   ├── taylor_couette_flow.py
├── LICENSE.txt
├── manuscript
│   ├── model6-num-names.bst
│   ├── paper.tex
│   └── references.bib
├── README.md
├── requirements.txt
└── vis_3d.py
```

The algorithm mentioned in the manuscript is implemented in
`particle_packing.py`, `colagrossi.py` and `jiang.py`. The `sort_points.py`
is a standalone code used to get sorted points of a 2d curve from random
points. The `data/` folder contains all the inputs used in the research.

The following files contain the test problems used in the manuscript:
- `case_2d_periodic.py` - Packing stability in 2D
- `case_3d_bounded_lattice.py` - Packing stability in 3D
- `case_2d_bounded_cylinder.py` - Circular cylinder 
- `case_2d_bounded_airfoil.py` and `case_2d_bounded_object.py` -  Packing at different resolutions 
- `casa_3d_bounded_sphere.py` and `case_3d_bunny.py` - Packing in 3D 

## Installation

This requires pysph to be setup along with automan. See the
`requirements.txt`. To setup perform the following:

0. Setup a suitable Python distribution, using
   [edm](https://docs.enthought.com/edm/) or [conda](https://conda.io) or a
   [virtualenv](https://virtualenv.pypa.io/).

1. Clone this repository:
```
    $ git clone https://gitlab.com/pypr/sph_geom.git
```

2. Run the following from your Python environment:
```
    $ cd sph_geom
    $ pip install -r requirements.txt
```
## Running an example

A simple example to check if your installation proceeded as expected, run the
following command from the top level directory:

    $ python code/case_2d_bounded_cylinder.py --dx 0.1 --max-step 20000 --tf 200

it will take about 25 seconds on a single core CPU to run the above simulation and
the output will be generated in the current directory with a folder named
`case_2d_bounded_cylinder_outputs`.

## Packing for a new geometry

In the case of a 2D geometry, extract the boundary of the geometry in the
form of discrete points. For example see `code/data/starfish.csv`. The extracted points
can be sorted using the simple program `sort_points.py` using the following
command
    
    python sort_points.py starfish.csv

This will generate a file by the name `starfish_sorted.csv`. Please note this
only handles a single non-self-intersecting curve.

Now, the generated file path can be replaced in the code
`code/case_2d_bounded_object.py:37`. Make sure to change the domain size i.e
`self.L` and `self.B` appropriately.

Then, run the following command to generate the packing

    cd code
    python case_2d_bounded_object.py --tf 200 --max-step 20000 --openmp

In the case of a 3D geometry, the geometry is required in a STL file format.
Although the code has features to correctly pull the surface by dx/2. We
recommend the user to make sure that the STL file is a surface on which the
boundary particles are desired.

In order or pack around a new geometry, replace the file name in
`code/case_3d_bunny.py:37` with the new filename. Make sure to change the
domain size accordingly.

Then, run the following command to generate the packing

    cd code
    python case_3d_bunny.py --tf 200 --max-step 20000 --openmp

## Visualization

The output file can be viewed in the PySPH viewer using the following command

    pysph view folder_name

If you wish to use another visualization tool, the output files can be
converted into VTK files using the command

    pysph dump_vtk folder_name -d vtk_folder_name

However, if the output folder is not passed, then the VTK files are dumped in
the input directory.

## Reproducing the results of the paper

The paper and the results are all automated using the
[automan](https://automan.readthedocs.io) package which should already be
installed as part of the above installation. This will perform all the required
simulations (this can take a while) and also generate all the plots for the
manuscript. For the plots made using `vis_3d.py`,
[paraview](https://www.paraview.org/download/) is required to be installed in
the system and its `bin` path be added to `PATH` environment variable.

To use the automation code, do the following::

    $ python automate.py
    # or
    $ ./automate.py

By default the simulation outputs are in the ``outputs`` directory and the
final plots for the paper are in ``manuscript/figures``.

## Building the pdf

The manuscript is written with LaTeX and if you have that installed you may do
the following:

```
$ cd manuscript
$ pdflatex paper.tex
$ bibtex paper
$ pdflatex paper.tex
$ pdflatex paper.tex
```
