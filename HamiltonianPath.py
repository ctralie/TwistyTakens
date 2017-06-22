import networkx as nx
import networkx.algorithms.matching as matching
import numpy as np
import matplotlib.pyplot as plt
from Geom3D.PolyMesh import *

def getHamiltonianPath(mesh):
    #Step 1: Build dual graph
    G =  nx.Graph()
    faces = mesh.faces
    V = [f.getCentroid().tolist() for f in faces]
    V = np.array(V)
    #Add face edges
    for f in faces:
        for f2 in f.getAdjacentFaces():
            G.add_edge(f.ID, f2.ID)
    #TODO: Deal with meshes with boundary

    eAll = set(G.edges())
    eMatched = matching.maximal_matching(G)
    eUnmatched = eAll - eMatched
