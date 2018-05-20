import networkx as nx
import networkx.algorithms.matching as matching
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from Geom3D.PolyMesh import *
import os

def getBlossom5PerfectMatching(G):
    """
    Wrap around the Blossom 5 C++ library for computing a perfect matching
    :param G: A networkx graph
    """
    edges = G.edges()
    fout = open("edges.txt", "w")
    fout.write("%i %i\n"%(len(G.nodes()), len(edges)))
    for e in edges:
        fout.write("%i %i 1\n"%e)
    fout.close()

    subprocess.call(["./blossom5", "-e", "edges.txt", "-w", "edgesres.txt"])
    fin = open("edgesres.txt")
    res = fin.readlines()
    res = res[1::]
    for i in range(len(res)):
        idxs = res[i].split()
        res[i] = [int(idx) for idx in idxs]
    fin.close()
    os.remove("edges.txt")
    os.remove("edgesres.txt")
    return res

def getSubdividedVertices(mesh, Vs, v1, v2, u):
    aPos = 0.7*mesh.VPos[v1.ID, :] + 0.15*mesh.VPos[v2.ID, :] + 0.15*mesh.VPos[u.ID, :]
    a = len(Vs)
    Vs.append(aPos)
    bPos = 0.15*mesh.VPos[v1.ID, :] + 0.7*mesh.VPos[v2.ID, :] + 0.15*mesh.VPos[u.ID, :]
    b = len(Vs)
    Vs.append(bPos)
    return (a, b)

def getOppositeVertex(f, e):
    """
    Get vertex on the opposite side of a triangle edge
    """
    [v1, v2] = [e.v1, e.v2]
    e1 = (set(f.edges) - set([e])).pop()
    u = ((set([v1, v2]) | set([e1.v1, e1.v2])) - set([v1, v2])).pop()
    return u

def getRotatedCycle(cycle, i):
    """
    Re-index cycle list so the vertices involved in the matched
    edge appear first
    """
    idx = cycle.index(i)
    c = np.array(cycle)
    return np.roll(c, -idx).tolist()

def getHamiltonianPath(mesh):
    """
    For now, assumes a triangle mesh without boundary
        #TODO: Deal with meshes with boundary
    """
    ## Step 1: Build dual graph
    G =  nx.Graph()
    faces = mesh.faces
    Vs = [f.getCentroid() for f in faces]
    #Add face edges
    for f in faces:
        for f2 in f.getAdjacentFaces():
            G.add_edge(f.ID, f2.ID)

    ## Step 2: Perform a perfect matching on the dual graph
    eAll = set(G.edges())
    getBlossom5PerfectMatching(G)
    print "Doing matching..."
    eMatched = getBlossom5PerfectMatching(G)
    print "Finished matching"
    G.remove_edges_from(eMatched)


    ## Step 3: Extract cycles from the perfect matching
    cycles = nx.cycle_basis(G, 0)
    print "len(cycles) = ", len(cycles)
    #Make sure everything got matched
    repped = set([])
    for c in cycles:
        for i in c:
            repped.add(i)
    if not (len(faces) == len(repped)):
        print "Warning: Not all faces represented in perfect matching: %i of %i"%(len(repped), len(faces))


    ## Step 4: Find MST of cycles
    #First create a dictionary that goes from vertex to cycle ID
    cycleIDs = {}
    for i in range(len(cycles)):
        c = cycles[i]
        for fid in c:
            cycleIDs[fid] = i
    #Now connect cycles with matched edges
    cycleGraph = nx.Graph()
    for (v1, v2) in eMatched:
        cycleGraph.add_edge(cycleIDs[v1], cycleIDs[v2], f1 = v1, f2 = v2)
    #Find a spanning tree of cycles
    T = nx.minimum_spanning_tree(cycleGraph)
    print "len(T.edges) = ", len(T.edges())

    ## Step 5: Use the spanning tree to connect all cycles into one big cycle
    cycleRet = None
    for (cid1, cid2) in T.edges():
        f1 = mesh.faces[T[cid1][cid2]['f1']]
        f2 = mesh.faces[T[cid1][cid2]['f2']]
        [cycle1, cycle2] = [cycles[cycleIDs[f1.ID]], cycles[cycleIDs[f2.ID]]]
        #Rotate cycles for convenience
        cycle1 = getRotatedCycle(cycle1, f1.ID)
        cycle2 = getRotatedCycle(cycle2, f2.ID)

        e = getFacesEdgeInCommon(f1, f2)
        if not e:
            print "Error: Edge not in common between two adjacent faces"
            continue
        [v1, v2] = [e.v1, e.v2]
        #Get opposite vertices
        u = getOppositeVertex(f1, e)
        w = getOppositeVertex(f2, e)
        #Setup internal subdivided vertices
        (a, b) = getSubdividedVertices(mesh, Vs, v1, v2, u)
        (c, d) = getSubdividedVertices(mesh, Vs, v1, v2, w)
        #Merge the two to make a new big cycle
        newCycle = cycle1 + [a, c] + cycle2 + [d, b]
        cycles[cid1] = newCycle
        cycles[cid2] = newCycle
        cycleRet = newCycle
    for c in cycles:
        print len(c), " ",
    print ""
    print "len(cycle) = ", len(cycleRet)
    print "len(faces) = ", len(faces)
    print "(len(cycles)-1)*2 + len(faces) = ", (len(cycles)-1)*2 + len(faces)

    Vs = np.array(Vs)
    return [Vs[cycleRet, :]]
