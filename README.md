<div align="center">    
 
# LFDeep: Multitask Learning of Biophysically-Detailed Neuron Models
   
</div>

## Project Structure

The project is organized as follows:

- **configs**: Contains configuration files for your experiments.
- **project**: The main project folder containing the deep leanring models and supporting classes.
- **environment.yml**: Defines the project's environment and dependencies.
- **main.py**: The main entry point for the project. You can run this file to execute the project.

## Concept

The human brain operates at multiple levels, from molecules to circuits, and understanding these complex processes requires integrated research efforts. Simulating biophysically-detailed neuron models is a computationally expensive but effective method for studying local neural circuits. Recent innovations have shown that artificial neural networks (ANNs) can accurately predict the behavior of these detailed models in terms of spikes, electrical potentials, and optical readouts. While these methods have the potential to accelerate large network simulations by several orders of magnitude compared to conventional differential equation-based modeling, they currently only predict voltage outputs for the soma or a select few neuron compartments. Our novel approach, based on enhanced versions of state-of-the-art architectures for multitask learning, allows for the simultaneous prediction of membrane potentials in each compartment of a neuron model, at a speed of up to two orders of magnitude faster than classical simulation methods. By predicting all membrane potentials together, our approach not only allows for a comparison of model output with a wider range of experimental recordings, but it also provides the first stepping stone towards predicting local field potentials. 

## How to Run

First, install the dependencies and clone the project:

```bash
# Clone the project
git clone https://github.com/Jonas-Verhellen/LFDeep

# Navigate to the project directory
cd LFDeep

# Create and activate the conda environment
conda env create -f environment.yml
conda activate lfdeep

# run the main file
python main.py
```

## Authors

* Jonas Verhellen
* Kosio Beshkov
* Sebastian Amundsen
* Torbjørn V. Ness
* Gaute T. Einevoll

## Affiliations

* Center for Integrative Neuroplasticity, University of Oslo
* Department of Physics, Norwegian University of Life Sciences
