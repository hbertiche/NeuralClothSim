## Neural Cloth Simulation

<img src="https://raw.githubusercontent.com/hbertiche/hbertiche.github.io/main/imgs/publications/NCS.png">

<a href="hbertiche.github.io/NeuralClothSim">Project Page</a> | <a href="https://dl.acm.org/doi/10.1145/3550454.3555491">Paper</a> | <a href="">arXiv</a> | <a href="https://youtu.be/6HxXLBzRXFg">Video 1</a> | <a href="https://youtu.be/iRbNlQHwbbA">Video 2</a>

### Abstract

>
>
>We present a general framework for the garment animation problem through unsupervised deep learning inspired in physically based simulation. Existing trends in the literature already explore this possibility. Nonetheless, these approaches do not handle cloth dynamics. Here, we propose the first methodology able to learn realistic cloth dynamics unsupervisedly, and henceforth, a general formulation for neural cloth simulation. The key to achieve this is to adapt an existing optimization scheme for motion from simulation based methodologies to deep learning. Then, analyzing the nature of the problem, we devise an architecture able to automatically disentangle static and dynamic cloth subspaces by design. We will show how this improves model performance. Additionally, this opens the possibility of a novel motion augmentation technique that greatly improves generalization. Finally, we show it also allows to control the level of motion in the predictions. This is a useful, never seen before, tool for artists. We provide of detailed analysis of the problem to establish the bases of neural cloth simulation and guide future research into the specifics of this domain.

<a href="mailto:hugo_bertiche@hotmail.com">Hugo Bertiche</a>, <a href="mailto:mmadadi@cvc.uab.cat">Meysam Madadi</a> and <a href="https://sergioescalera.com/">Sergio Escalera</a>

## Run the code

### Train

To train a model go to `ncs/` and run:
```
python train.py --config [CONFIG_FILE] --gpu_id [GPU_IDs]
```
`[CONFIG_FILE]` is a path pointing to a JSON file with the configuration (See below for an explanation). There are two sample configuration files in `ncs/configs/`.<br>
`[GPU_IDs]` is the IDs of the GPUs you want to use for training (e.g.: `0,1`).
<br>

During training, collision detection takes a significant amount of time. We allow parallelization by using <a href="https://www.ray.io/">Ray</a>. To do so, go to `ncs/model/ncs.py` line 102, where the collision detection layer is instanced and set `use_ray=True`. While this provides a significant boost in performance, we observed it might malfunction in some machines/OS.

### Predict

To obtain predictions go to `ncs/` and run:
```
python predict.py --config [CONFIG_FILE] --gpu_id [GPU_ID]
```
In this case, using multiple GPUs is not supported (no need to).

### Visualization

To view predictions, from the root folder, run:
```
blender --python view.py -- [CONFIG_FILE]
```
Blender is required (Tested in Blender 3.0).

## Body models and garments

The bodies used for simulation must be placed in `body_models/[BODY_NAME]/`. Garments are usually specific for a given body, and thus are expected to be within the corresponding body model folder. See the sample bodies and garments provided as examples.

Within `body_models/` you will also find JSON files defining body skeleton metadata. See the examples provided.

## Data

To neurally simulate cloth on top of a 3D character, we need pose sequence data.<br>
Pose sequence datasets should be placed in `data/[DATASET_NAME]/`.<br>
See the toy datasets provided as examples.

## Configuration files

To allow experimentation using different 3D bodies, garments, fabrics, etc. we define the configuration of the experiments as JSON files. Next we describe the structure of these JSON files. We additionally recommend looking at the provided configuration files as examples.

### Experiment

`epochs`: number of epochs.<br>
`batch_size`: batch size, the bigger the better.<br>
`learning_rate`: learning rate.<br>
`temporal_window_size`: length (in seconds) of the input sequences used for training.<br>
`reflect_probability`: data augmentation probability. This augmentation reflects the body pose (mirror).<br>
`motion_augmentation`: ratio of samples within a batch for which to apply motion augmentation (See paper, Sec. 4.4).<br>
`checkpoint`: if not `null`, it will load the defined checkpoint. It will look for checkpoints as `checkpoints/[CHECKPOINT]`<br>

### Body

`model`: name of the body model. It will use it to locate the body as `body_models/[BODY_NAME]/`.<br>
`skeleton`: name of the body skeleton JSON file. It will use it as `body_models/[SKELETON_NAME]_skeleton.json`.<br>
`input_joints`: list of indices of joints used as input of the network. This is useful to omit unnecessary joints.<br> 

### Garment

`name`: garment name. It will use it to locate the garment as `body_models/[BODY_NAME]/[GARMENT_NAME].obj`.<br>
`blend_weights_smoothing_iterations`: smooths blend weights. Blend weights are transferred from the body by proximity, this can be noisy. This is also very helpful for skirts (with 50-100 iterations).<br>

### Data

`dataset`: name of the dataset. It will use it to locate the dataset as `data/[DATASET_NAME]/`.<br>
`train`: name of a `.txt` file with the list of training sequences. It will look for the `.txt` in `ncs/dataset/txt/[DATASET_NAME]/[TXT_NAME]`.<br>
`validation`: list of validation sequences, used to compute metrics during training.<br>
`test`: list of test sequences, used to obtain predictions by the `ncs/predict.py` script.<br>
`fps`: FPS at which to run the model. This number can be chosen arbitrarily (we run our experiments with 30), it does not have to be the FPS of the pose sequences (pose sequences do not even need to have uniform FPS). Data will be sampled at the desired FPS.<br>

### Model

`blend_weights_optimize`: boolean value to allow optimization of the cloth blend weights.

### Loss

`cloth`: specify the cloth model configuration. The code supports three different formulations. Here are examples for each formulation:<br>
Mass spring:
```
{
  "type": "mass-spring",
  "edge": 10.0
}
```
[Baraff and Witkin 1998]
```
{
  "type": "baraff98",
  "stretch": 10.0,
  "shear": 1.0
}
```
Saint Venant Kirchhoff
```
{
  "type": "stvk",
  "lambda_": 20.9,
  "mu": 11.1
}
```

`bending`: bending stiffness of the cloth.<br>
`collision_weight`: weight for the collision loss.<br>
`collision_threshold`: desired margin (in meters) between body and cloth. Epsilon in eq. 7 of the paper.<br>
`density`: fabric density in (kg/m<sup>2</sup>).<br>
`pinning`: weight for the pinning loss.<br>
`gravity`: gravity axis and sign (e.g.: `-Y`). Alternatively, an arbitrary 3-dimensional array can be provided (e.g.: `[4.8, -5.21, 1.0]`).<br>

## Citation
```
@article{10.1145/3550454.3555491,
        author = {Bertiche, Hugo and Madadi, Meysam and Escalera, Sergio},
        title = {Neural Cloth Simulation},
        year = {2022},
        issue_date = {December 2022},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {41},
        number = {6},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3550454.3555491},
        doi = {10.1145/3550454.3555491},
        abstract = {We present a general framework for the garment animation problem through unsupervised deep learning inspired in physically based simulation. Existing trends in the literature already explore this possibility. Nonetheless, these approaches do not handle cloth dynamics. Here, we propose the first methodology able to learn realistic cloth dynamics unsupervisedly, and henceforth, a general formulation for neural cloth simulation. The key to achieve this is to adapt an existing optimization scheme for motion from simulation based methodologies to deep learning. Then, analyzing the nature of the problem, we devise an architecture able to automatically disentangle static and dynamic cloth subspaces by design. We will show how this improves model performance. Additionally, this opens the possibility of a novel motion augmentation technique that greatly improves generalization. Finally, we show it also allows to control the level of motion in the predictions. This is a useful, never seen before, tool for artists. We provide of detailed analysis of the problem to establish the bases of neural cloth simulation and guide future research into the specifics of this domain.},
        journal = {ACM Trans. Graph.},
        month = {nov},
        articleno = {220},
        numpages = {14},
        keywords = {simulation, neural network, dynamics, disentangle, deep learning, cloth, unsupervised}
}
```
