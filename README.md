# EmbodiedQA

Code for the paper

**[Embodied Question Answering][1]**  
Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, Dhruv Batra  
[arxiv.org/abs/1711.11543][2]  
CVPR 2018 (Oral)

In Embodied Question Answering (EmbodiedQA), an agent is spawned at a random location in a 3D environment and asked a question (for e.g. "What color is the car?"). In order to answer, the agent must first intelligently navigate to explore the environment, gather necessary visual information through first-person vision, and then answer the question ("orange").

![](https://i.imgur.com/jeI7bxm.jpg)

This repository provides

- [Pretrained CNN](#pretrained-cnn) for [House3D][house3d]
- Code for [generating EQA questions](#question-generation)
    - EQA v1: location, color, place preposition
    - EQA v1-extended: existence, logical, object counts, room counts, distance comparison
- Code to train and evaluate [navigation](#navigation) and [question-answering](#visual-question-answering) models
    - [independently with supervised learning](#supervised-learning) on shortest paths
    - jointly using [reinforcement learning](#reinforce)

If you find this code useful, consider citing our work:

```
@inproceedings{embodiedqa,
  title={{E}mbodied {Q}uestion {A}nswering},
  author={Abhishek Das and Samyak Datta and Georgia Gkioxari and Stefan Lee and Devi Parikh and Dhruv Batra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

## Setup

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

Download the [SUNCG v1 dataset](https://github.com/facebookresearch/House3D/blob/master/INSTRUCTION.md#usage-instructions) and [install House3D](https://github.com/abhshkdz/House3D/tree/master/renderer#rendering-code-of-house3d).

NOTE: This code uses a [fork of House3D](https://github.com/abhshkdz/house3d) with a few changes to support arbitrary map discretization resolutions.

## Question generation

Questions for EmbodiedQA are generated programmatically, in a manner similar to [CLEVR (Johnson et al., 2017)][clevr].

NOTE: Pre-generated EQA v1 questions are available for download [here][eqav1].

### Generating questions for all templates in EQA v1, v1-extended

```
cd data/question-gen
./run_me.sh MM_DD
```

### List defined question templates

```
from engine import Engine

E = Engine()
for i in E.template_defs:
    print(i, E.template_defs[i])
```

### Generate questions for a particular template (say `location`)

```
from house_parse import HouseParse
from engine import Engine

Hp = HouseParse(dataDir='/path/to/suncg')
Hp.parse('0aa5e04f06a805881285402096eac723')

E = Engine()
E.cacheHouse(Hp)
qns = E.executeFn(E.template_defs['location'])

print(qns[0]['question'], qns[0]['answer'])
# what room is the clock located in? bedroom

```

## Pretrained CNN

We trained a shallow encoder-decoder CNN from scratch in the House3D environment,
for RGB reconstruction, semantic segmentation and depth estimation.
Once trained, we throw away the decoders, and use the encoder as a frozen feature
extractor for navigation and question answering. The CNN is available for download here:

`wget https://www.dropbox.com/s/ju1zw4iipxlj966/03_13_h3d_hybrid_cnn.pt`

The training code expects the checkpoint to be present in `training/models/`.

## Supervised Learning

### Download and preprocess the dataset

Download [EQA v1][eqav1] and shortest path navigations:

```
wget https://www.dropbox.com/s/6zu1b1jzl0qt7t1/eqa_v1.json
wget https://www.dropbox.com/s/lhajthx7cdlnhns/a-star-500.zip
unzip a-star-500.zip
```

If this is the first time you are using SUNCG, you will have to clone and use the
[SUNCG toolbox](https://github.com/shurans/SUNCGtoolbox#convert-to-objmtl)
to generate obj + mtl files for the houses in EQA.

NOTE: Shortest paths have been updated.  Earlier we computed shortest paths using a discrete grid world, but we found that these shortest paths were sometimes innacurate.  Old shortest paths are [here](https://www.dropbox.com/s/vgp2ygh1bht1jyb/shortest-paths.zip).

```
cd utils
python make_houses.py \
    -eqa_path /path/to/eqa.json \
    -suncg_toolbox_path /path/to/SUNCGtoolbox \
    -suncg_data_path /path/to/suncg/data_root
```

Preprocess the dataset for training


```
cd training
python utils/preprocess_questions_pkl.py \
    -input_json /path/to/eqa_v1.json \
    -shortest_path_dir /path/to/shortest/paths/a-star-500 \
    -output_train_h5 data/train.h5 \
    -output_val_h5 data/val.h5 \
    -output_test_h5 data/test.h5 \
    -output_data_json data/data.json \
    -output_vocab data/vocab.json
```

### Visual question answering

Update pretrained CNN path in `models.py`.

`python train_vqa.py -input_type ques,image -identifier ques-image -log -cache`

This model computes question-conditioned attention over last 5 frames from oracle navigation (shortest paths),
and predicts answer. Assuming shortest paths are optimal for answering the question -- which is predominantly
true for most questions in EQA v1 (`location`, `color`, `place preposition`) with the
exception of a few `location` questions that might need more visual context than walking right up till the object --
this can be thought of as an upper bound on expected accuracy, and performance will get worse when navigation
trajectories are sampled from trained policies.

A pretrained VQA model is available for download [here](https://www.dropbox.com/s/jd15af00r7m8neh/vqa_11_18_2018_va0.6154.pt). This gets a top-1 accuracy of 61.54% on val, and 58.46% on test (with GT navigation).

Note that keeping the `cache` flag ON caches images as they are rendered in the first training epoch, so that subsequent epochs are very fast. This is memory-intensive though, and consumes ~100-120G RAM.

### Navigation

Download potential maps for evaluating navigation and training with REINFORCE.

```
wget https://www.dropbox.com/s/53edqtr04jts4q0/target-obj-conn-maps-500.zip
```

#### Planner-controller policy

`python train_nav.py -model_type pacman -identifier pacman -log`

## REINFORCE

```
python train_eqa.py \
    -nav_checkpoint_path /path/to/nav/ques-image-pacman/checkpoint.pt \
    -ans_checkpoint_path /path/to/vqa/ques-image/checkpoint.pt \
    -identifier ques-image-eqa \
    -log
```

## Changelog

### 09/07

- We added the baseline models from the CVPR paper (Reactive and LSTM).
- With the LSTM model, we achieved d_T values of: 0.74693/3.99891/8.10669 on the test set for d equal to 10/30/50 respectively training with behavior cloning (no reinforcement learning).
- We also updated the shortest paths to fix an issue with the shortest path algorithm we initially used.  Code to generate shortest paths is [here](https://github.com/facebookresearch/EmbodiedQA/blob/master/data/shortest-path-gen/generate-paths-a-star.py).

### 06/13

This code release contains the following changes over the CVPR version

- Larger dataset of questions + shortest paths
- Color names as answers to color questions (earlier they were hex strings)

## Acknowledgements

- Parts of this code are adapted from [pytorch-a3c][pytorch-a3c] by Ilya Kostrikov
- [Lisa Anne Hendricks](https://people.eecs.berkeley.edu/~lisa_anne/) and [Licheng Yu](http://www.cs.unc.edu/~licheng/)
helped with running / testing / debugging code prior to release

## License

BSD

[1]: https://embodiedqa.org
[2]: https://arxiv.org/abs/1711.11543
[house3d]: https://github.com/facebookresearch/house3d
[dijkstar]: https://bitbucket.org/wyatt/dijkstar
[pytorch-a3c]: https://github.com/ikostrikov/pytorch-a3c
[eqav1]: https://embodiedqa.org/data
[clevr]: https://github.com/facebookresearch/clevr-dataset-gen
