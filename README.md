# UniGarmentManip: A <u>**Uni**</u>fied Framework for Category-Level <u>**GarmentManip**</u>ulation via Dense Visual Correspondence

![Overview](./image/teaser.png)

## Introduction
Garment manipulation (**e.g.**, unfolding, folding and hanging clothes) is essential for future robots to accomplish home-assistant tasks, while highly challenging due to the diversity of garment configurations, geometries and deformations.
Although able to manipulate similar shaped garments in a certain task,
previous works mostly have to design different policies for different tasks, could not generalize to garments with diverse geometries, and often rely heavily on human-annotated data.
In this paper, we leverage the property that,
garments in a certain category have similar structures,
and then learn the topological dense (point-level) visual correspondence among garments in the category level with different deformations in the self-supervised manner. 
The topological correspondence can be easily adapted to the functional correspondence to guide the manipulation policies for various downstream tasks,
within only one or few-shot demonstrations.  
Experiments over garments in 3 different categories on 3 representative tasks in diverse scenarios, 
using one or two arms, 
taking one or more steps,  
inputting flat or messy garments,
demonstrate the effectiveness of our proposed method.

## Structure of the Repository
This repository provides data and code as follows.
```
    garmentgym/             # The garment manipulation simulator
    skeleton/               # The skeleton code for garment structure learning
    task/                   # The code for garment manipulation tasks
    train/                  # The code for training dense visual correspondence
    collect/                # The code for collecting data
    demonstration/          # The code for one-shot demonstration
```
You can follow the README in EACH FOLDER to install and run the code.
