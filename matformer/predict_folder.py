"""Module to predict using a trained model."""
import torch
import csv
import os
import sys
import time
from jarvis.core.atoms import Atoms
from matformer.data import get_train_val_loaders
from matformer.train import train_dgl
from matformer.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from matformer import models
from matformer.data import get_train_val_loaders
from matformer.config import TrainingConfig
from matformer.models.pyg_att import Matformer
import argparse
import json
import glob
import pandas as pd
structures_folder = '/home/abshe/Matformer/my_materials/deformed_materials/'
config_file = '/home/abshe/Matformer/matformer/logs/my_materials_pristine/config.json'
config = json.load(open(config_file))
id_prop = pd.read_csv('/home/abshe/Matformer/my_materials/deformed_materials/id_prop.csv',index_col=0,header=None)
device = "cpu"
# if torch.cuda.is_available():
#     device = torch.device("cuda")
_model = {
        "matformer" : Matformer,
    }

config = TrainingConfig(**config)
net = _model.get(config.model.name)(config.model)
print(net)

net.to(device)
checkpoint_tmp = torch.load('/home/abshe/Matformer/matformer/logs/my_materials_pristine/best_model_5130_neg_mae=-0.0743.pt',map_location=torch.device('cpu'))
to_load = {
    "model": net,
    # "optimizer": optimizer,
    # "lr_scheduler": scheduler,
    # "trainer": trainer,
}
Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
net.eval()
targets = []
predictions = []
import time
t1 = time.time()

comparison = []
from  matformer.graphs import PygGraph
MAE = 0
count = 0
for i,r in id_prop.iterrows():
    structure = Atoms.from_poscar(structures_folder + i)
    print(torch.tensor(r.to_numpy()))
    g, lg = PygGraph.atom_dgl_multigraph(structure)
    print(g)
    out_data = (
        net([g.to(device), lg.to(device), None])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()[0]
    )
    comparison += [[i, r.values[0],out_data]]
    MAE += abs(r.values[0]-out_data)
    count += 1


    # cvn = Spacegroup3D(structure).conventional_standard_structure
    # g, lg = Graph.atom_dgl_multigraph(cvn)
    # out_data = (
    #     model([g.to(device), lg.to(device)])
    #     .detach()
    #     .cpu()
    #     .numpy()
    #     .flatten()
    #     .tolist()[0]
    # )
    # print("cvn", out_data)


    # atoms = atoms.make_supercell([3, 3, 3])
    # g, lg = Graph.atom_dgl_multigraph(atoms)
    # out_data = (
    #     model([g.to(device), lg.to(device)])
    #     .detach()
    #     .cpu()
    #     .numpy()
    #     .flatten()
    #     .tolist()[0]
    # )
    # print("supercell", out_data)
t2 = time.time()

comparison = pd.DataFrame(comparison)
comparison.to_csv('comparison.csv')
print(MAE/count)