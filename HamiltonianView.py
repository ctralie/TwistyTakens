#Based off of http://wiki.wxpython.org/GLCanvas
#Lots of help from http://wiki.wxpython.org/Getting%20Started
from OpenGL.GL import *
from OpenGL.arrays import vbo
import wx
from wx import glcanvas

from Geom3D.Primitives3D import *
from Geom3D.PolyMesh import *
from Geom3D.Cameras3D import *
from Geom3D.MeshCanvas import *
from HamiltonianPath import *
from struct import *
from sys import exit, argv
import random
import numpy as np
import scipy.io as sio
from pylab import cm
import os
import math
import time
from time import sleep
from pylab import cm
import matplotlib.pyplot as plt
from OpenGL.arrays import vbo

class MeshViewerCanvas(BasicMeshCanvas):
    def __init__(self, parent):
        super(MeshViewerCanvas, self).__init__(parent)
        self.hamVBO = None
        self.hamPath = np.zeros((0, 3))
        self.displayHamPath = True
        self.displayMeshVertices = False

    def displayMeshFacesCheckbox(self, evt):
        self.displayMeshFaces = evt.Checked()
        self.Refresh()

    def displayMeshEdgesCheckbox(self, evt):
        self.displayMeshEdges = evt.Checked()
        self.Refresh()

    def displayBoundaryCheckbox(self, evt):
        self.displayBoundary = evt.Checked()
        self.Refresh()

    def displayMeshVerticesCheckbox(self, evt):
        self.displayMeshVertices = evt.Checked()
        self.Refresh()

    def displayVertexNormalsCheckbox(self, evt):
        self.displayVertexNormals = evt.Checked()
        self.Refresh()

    def displayFaceNormalsCheckbox(self, evt):
        self.displayFaceNormals = evt.Checked()
        self.Refresh()

    def useLightingCheckbox(self, evt):
        self.useLighting = evt.Checked()
        self.Refresh()

    def useTextureCheckbox(self, evt):
        self.useTexture = evt.Checked()
        self.Refresh()

    def displayHamCheckbox(self, evt):
        self.displayHamPath = evt.Checked()
        self.Refresh()

    def drawMeshStandard(self):
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
        self.camera.gotoCameraFrame()
        glLightfv(GL_LIGHT0, GL_POSITION, np.array([0, 0, 0, 1]))
        self.mesh.renderGL(self.displayMeshEdges, self.displayMeshVertices, self.displayMeshFaces, self.displayVertexNormals, self.displayFaceNormals, self.useLighting, self.useTexture, self.displayBoundary)

    def drawHamPath(self, N):
        """
        Draw hamiltonian path with magenta lines
        :param N: Draw the first N points along the path
        """
        if not self.hamVBO:
            return
        glEnableClientState(GL_VERTEX_ARRAY)
        self.hamVBO.bind()
        glVertexPointerf(self.hamVBO)
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        glColor3f(1.0, 0, 1.0)
        glDrawArrays(GL_LINES, 0, N)
        glPointSize(10)
        glColor3f(0.9, 0, 0.9)
        glDrawArrays(GL_POINTS, 0, N)
        self.hamVBO.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

    def repaint(self):
        self.setupPerspectiveMatrix()

        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.mesh:
            self.drawMeshStandard()

        if self.displayHamPath:
            self.drawHamPath(self.hamPath.shape[0])

        self.SwapBuffers()

    def OnComputeHamiltonianPath(self, evt):
        if self.hamVBO:
            self.hamVBO.delete()
        cycles = getHamiltonianPath(self.mesh)
        V = np.zeros((0, 3))
        print "len(cycles) = ", len(cycles)
        for c in cycles:
            v = np.zeros((c.shape[0]*2, 3))
            v[0::2, :] = c
            v[1:-1:2, :] = c[1::, :]
            v[-1, :] = c[0, :]
            V = np.concatenate((V, v), 0)
        self.hamPath = V
        self.hamVBO = vbo.VBO(np.array(V, dtype = np.float32))

class MeshViewerFrame(wx.Frame):
    (ID_LOADDATASET, ID_SAVEDATASET, ID_SAVEDATASETMETERS, ID_SAVESCREENSHOT, ID_CONNECTEDCOMPONENTS, ID_SPLITFACES, ID_TRUNCATE, ID_FILLHOLES, ID_GEODESICDISTANCES, ID_PRST, ID_INTERPOLATECOLORS, ID_SAVEROTATINGSCREENSOTS, ID_SAVELIGHTINGSCREENSHOTS, ID_COMPUTEHAMILTONIANPATH, ID_CLEARLAPLACEVERTICES, ID_SOLVEWITHCONSTRAINTS, ID_MEMBRANEWITHCONSTRAINTS, ID_GETHKS, ID_GETHEATFLOW) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)

    def __init__(self, parent, id, title, pos=DEFAULT_POS, size=DEFAULT_SIZE, style=wx.DEFAULT_FRAME_STYLE, name = 'GLWindow'):
        style = style | wx.NO_FULL_REPAINT_ON_RESIZE
        super(MeshViewerFrame, self).__init__(parent, id, title, pos, size, style, name)
        #Initialize the menu
        self.CreateStatusBar()

        self.size = size
        self.pos = pos
        print "MeshViewerFrameSize = %s, pos = %s"%(self.size, self.pos)
        self.glcanvas = MeshViewerCanvas(self)

        #####File menu
        filemenu = wx.Menu()
        menuOpenMesh = filemenu.Append(MeshViewerFrame.ID_LOADDATASET, "&Load Mesh","Load a polygon mesh")
        self.Bind(wx.EVT_MENU, self.OnLoadMesh, menuOpenMesh)
        menuSaveMesh = filemenu.Append(MeshViewerFrame.ID_SAVEDATASET, "&Save Mesh", "Save the edited polygon mesh")
        self.Bind(wx.EVT_MENU, self.OnSaveMesh, menuSaveMesh)
        menuSaveScreenshot = filemenu.Append(MeshViewerFrame.ID_SAVESCREENSHOT, "&Save Screenshot", "Save a screenshot of the GL Canvas")
        self.Bind(wx.EVT_MENU, self.OnSaveScreenshot, menuSaveScreenshot)

        menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

        #####Hamiltonian Path menu
        hammenu = wx.Menu()
        menuComputeHam = hammenu.Append(MeshViewerFrame.ID_COMPUTEHAMILTONIANPATH, "&Compute Hamiltonian Path", "Compute Hamiltonian Path")
        self.Bind(wx.EVT_MENU, self.glcanvas.OnComputeHamiltonianPath, menuComputeHam)


        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        menuBar.Append(hammenu, "&Hamiltonian")
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

        self.rightPanel = wx.BoxSizer(wx.VERTICAL)

        #Buttons to go to a default view
        viewPanel = wx.BoxSizer(wx.HORIZONTAL)
        topViewButton = wx.Button(self, -1, "Top")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewFromTop, topViewButton)
        viewPanel.Add(topViewButton, 0, wx.EXPAND)
        sideViewButton = wx.Button(self, -1, "Side")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewFromSide, sideViewButton)
        viewPanel.Add(sideViewButton, 0, wx.EXPAND)
        frontViewButton = wx.Button(self, -1, "Front")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewFromFront, frontViewButton)
        viewPanel.Add(frontViewButton, 0, wx.EXPAND)
        self.rightPanel.Add(wx.StaticText(self, label="Views"), 0, wx.EXPAND)
        self.rightPanel.Add(viewPanel, 0, wx.EXPAND)

        #Checkboxes for displaying data
        self.displayMeshFacesCheckbox = wx.CheckBox(self, label = "Display Mesh Faces")
        self.displayMeshFacesCheckbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshFacesCheckbox, self.displayMeshFacesCheckbox)
        self.rightPanel.Add(self.displayMeshFacesCheckbox, 0, wx.EXPAND)
        self.displayMeshEdgesCheckbox = wx.CheckBox(self, label = "Display Mesh Edges")
        self.displayMeshEdgesCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshEdgesCheckbox, self.displayMeshEdgesCheckbox)
        self.rightPanel.Add(self.displayMeshEdgesCheckbox, 0, wx.EXPAND)
        self.displayBoundaryCheckbox = wx.CheckBox(self, label = "Display Boundary")
        self.displayBoundaryCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayBoundaryCheckbox, self.displayBoundaryCheckbox)
        self.rightPanel.Add(self.displayBoundaryCheckbox, 0, wx.EXPAND)
        self.displayMeshVerticesCheckbox = wx.CheckBox(self, label = "Display Mesh Points")
        self.displayMeshVerticesCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshVerticesCheckbox, self.displayMeshVerticesCheckbox)
        self.rightPanel.Add(self.displayMeshVerticesCheckbox, 0, wx.EXPAND)
        self.displayVertexNormalsCheckbox = wx.CheckBox(self, label = "Display Vertex Normals")
        self.displayVertexNormalsCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayVertexNormalsCheckbox, self.displayVertexNormalsCheckbox)
        self.rightPanel.Add(self.displayVertexNormalsCheckbox, 0, wx.EXPAND)
        self.displayFaceNormalsCheckbox = wx.CheckBox(self, label = "Display Face Normals")
        self.displayFaceNormalsCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayFaceNormalsCheckbox, self.displayFaceNormalsCheckbox)
        self.rightPanel.Add(self.displayFaceNormalsCheckbox, 0, wx.EXPAND)
        self.useLightingCheckbox = wx.CheckBox(self, label = "Use Lighting")
        self.useLightingCheckbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.useLightingCheckbox, self.useLightingCheckbox)
        self.rightPanel.Add(self.useLightingCheckbox, 0, wx.EXPAND)
        self.useTextureCheckbox = wx.CheckBox(self, label = "Use Texture")
        self.useTextureCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.useTextureCheckbox, self.useTextureCheckbox)
        self.rightPanel.Add(self.useTextureCheckbox, 0, wx.EXPAND)
        self.displayHamCheckbox = wx.CheckBox(self, label = "Display Hamiltonian Path")
        self.displayHamCheckbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayHamCheckbox, self.displayHamCheckbox)
        self.rightPanel.Add(self.displayHamCheckbox, 0, wx.EXPAND)

        #Finally add the two main panels to the sizer
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        #cubecanvas = CubeCanvas(self)
        #self.sizer.Add(cubecanvas, 2, wx.EXPAND)
        self.sizer.Add(self.glcanvas, 2, wx.EXPAND)
        self.sizer.Add(self.rightPanel, 0, wx.EXPAND)

        self.SetSizer(self.sizer)
        self.Layout()
        self.glcanvas.Show()

    def OnLoadMesh(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "OFF files (*.off)|*.off|TOFF files (*.toff)|*.toff|OBJ files (*.obj)|*.obj", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            print dirname
            self.glcanvas.mesh = PolyMesh()
            print "Loading mesh %s..."%filename
            self.glcanvas.mesh.loadFile(filepath)
            self.glcanvas.meshCentroid = self.glcanvas.mesh.getCentroid()
            self.glcanvas.meshPrincipalAxes = self.glcanvas.mesh.getPrincipalAxes()
            print "Finished loading mesh"
            print self.glcanvas.mesh
            self.glcanvas.initMeshBBox()
            self.glcanvas.Refresh()
        dlg.Destroy()
        return

    def OnSaveMesh(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            self.glcanvas.mesh.saveFile(filepath, True)
            self.glcanvas.Refresh()
        dlg.Destroy()
        return

    def OnSaveScreenshot(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            saveImageGL(self.glcanvas, filepath)
        dlg.Destroy()
        return

    def OnExit(self, evt):
        self.Close(True)
        return

class MeshViewer(object):
    def __init__(self, filename = None, ts = False, sp = "", ra = 0):
        app = wx.App()
        frame = MeshViewerFrame(None, -1, 'MeshViewer')
        frame.Show(True)
        app.MainLoop()
        app.Destroy()

if __name__ == '__main__':
    viewer = MeshViewer()
