# Credits and Sources

All code in this repository is released under the MIT license (see LICENSE): the `mvb2r` tools and library,
the Blender scripts, the templates and the tests. Parts of it are ported from
[Blockout 2 Real](https://github.com/gokayfem/blockout-to-real) (same author, MIT): input ingestion, the JSON
scene spec of primitives, the doctor, the budgeted fal call layer, the handling of untrusted input, the
visibility-tested texel back-projection it extends, the test runner and the CI workflow. The models and tools
below are called or imported, not redistributed; each keeps its own licence and terms. Where a licence is not
stated here, read the model card or project page before relying on it.

## Models

| Model | Source | Used for | Licence / terms |
|---|---|---|---|
| Seedream 5.5 (edit and text-to-image) | ByteDance; `bytedance/seedream/v5.5/edit` and `bytedance/seedream/v5.5/text-to-image` on fal | the default image model: the multi-view sheet, per-view refinement, the render-then-refine round; in the first run also the fictional character reference | closed model; ByteDance's and fal's terms of use |
| Any other fal multi-reference edit endpoint | set in `case.json` `model` | the same stages, after `probe` | that model's terms |

## Methods

| Method | Source | Used for |
|---|---|---|
| Large-scale texturing of reconstructions | M. Waechter, N. Moehrle, M. Goesele, "Let There Be Color! Large-Scale Texturing of 3D Reconstructions", ECCV 2014 | view selection as a labelling problem with a seam cost, the photo-consistency check, global colour levelling and seam blending; adapted for generated views (`references/seams.md`) |
| Colour map optimisation | Q.-Y. Zhou, V. Koltun, "Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras", SIGGRAPH 2014 | the idea of correcting per-view colour and small misalignments against a shared texture, used in the levelling and the registration variant |
| Seamless image-based texture atlases | V. Lempitsky, D. Ivanov, "Seamless Mosaicing of Image-Based Texture Maps", CVPR 2007 | seam costs measured on both sides of a cut in the labelling methods |
| Graph-cut textures | V. Kwatra, A. Schödl, I. Essa, G. Turk, A. Bobick, "Graphcut Textures", SIGGRAPH 2003 | the difference-aware pairwise cost |
| Graph cuts | Y. Boykov, O. Veksler, R. Zabih, "Fast Approximate Energy Minimization via Graph Cuts", PAMI 2001; V. Kolmogorov, R. Zabih, "What Energy Functions Can Be Minimized via Graph Cuts?", PAMI 2004 | alpha-expansion for the labels |
| Multi-band blending | P. Burt, E. Adelson, "A Multiresolution Spline With Application to Image Mosaics", ACM TOG 1983 | the multi-band fusion variant |
| Visual hull | A. Laurentini, "The Visual Hull Concept for Silhouette-Based Image Understanding", PAMI 1994 | the silhouette-carving refit |
| GrabCut | C. Rother, V. Kolmogorov, A. Blake, SIGGRAPH 2004; OpenCV | generated object masks seeded from the clay object ids |
| ECC image alignment | G. Evangelidis, E. Psarakis, PAMI 2008; OpenCV | the second registration estimator in the acceptance rule |

## Tools

| Tool | Source | Used for | Licence |
|---|---|---|---|
| Blender, Cycles | Blender Foundation, Blender Python API (4.2+; developed on 5.0.1) | ingest, blockout, cameras, passes, clay and final renders, both `.blend` outputs | GPL-2.0-or-later |
| NumPy, SciPy, scikit-image | their projects | rasterisation, visibility, fusion, graph cuts (SciPy maximum flow), metrics | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | the OpenCV project | PNG and JPEG IO, masks, alignment | Apache-2.0 |
| Pillow | the Pillow project | image IO and contact sheets | HPND |
| fal-client | fal | every paid model call | see the project page |
| ffmpeg | the FFmpeg project | camera-move and tour videos (called as a program) | LGPL-2.1-or-later / GPL |
| Repository format | the sibling skills Video 2 GLSL (gokayfem/shader-skills), Video 2 Blender Nodes (gokayfem/video-to-blender), Video 2 Physics (gokayfem/video-to-physics), Video to CSS (gokayfem/video-to-css), Image 2 Sculpt (gokayfem/image-to-sculpt), Blockout 2 Real (gokayfem/blockout-to-real) | layout, adapters, plugin manifests | MIT |

## The first run's scene

No third-party asset is bundled in this repository. The scene of the first run, shown in `gallery/`, was built
from assets downloaded from their official hosts:

| Asset | Source | Licence |
|---|---|---|
| `Sofa_01` (geometry), `side_table_01` (geometry), `kloofendal_48d_partly_cloudy_puresky` HDRI | [Poly Haven](https://polyhaven.com) | CC0 |
| Human base mesh, clothes (`male_casualsuit02`, `shoes05`), hair envelope (`bob01`), eyes, pose (`milkman_sit_01a`) | [MakeHuman](https://www.makehumancommunity.org) system assets and poses01 packs (CC0 packs only), through the MPFB 2 add-on | assets CC0; MPFB add-on code GPL-3.0-or-later |

The person in the gallery is a fictional character generated with Seedream text-to-image for the run, not a
real individual. The photoreal views in `gallery/` were generated by Seedream 5.5 on fal; the clay views and
re-renders were rendered with Blender and Cycles.
