import sys
sys.path.append("Geom3D")
import numpy as np
import matplotlib.pyplot as plt
from PolyMesh import *
from Cameras3D import *
from MeshCanvas import *

SPHERE_RADIUS = 0.98

class SphereGLCanvas(BasicMeshCanvas):
    def __init__(self, parent, u, XTraj, C):
        BasicMeshCanvas.__init__(self, parent)
        self.u = u.flatten()
        self.C = C #Colors
        self.XTraj = XTraj
        #Initialize sphere mesh
        self.mesh = getSphereMesh(SPHERE_RADIUS, 6)
        #Initialize distance field colors
        d = np.arccos(self.mesh.VPos.dot(u))
        d = d-np.min(d)
        d = d/np.max(d)
        c = plt.get_cmap('gray')
        CSphere = c(np.array(np.round(255*d), dtype=np.int32))
        self.mesh.VColors = CSphere
        self.bbox = self.mesh.getBBox()
        self.camera.centerOnBBox(self.bbox, theta = math.pi/2, phi = math.pi/3)
        self.mesh2 = getSphereMesh(0.03, 3)
        self.mesh2.VPos += self.u
        self.mesh2.VColors = np.zeros_like(self.mesh2.VPos)
        self.mesh2.VColors[:, 0] = 1.0

        self.Refresh()
    
    def setupPerspectiveMatrix(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(180.0*self.camera.yfov/M_PI, float(self.size.x)/self.size.y, 0.001, 100)
    
    def repaint(self):
        self.setupPerspectiveMatrix()
        glClearColor(0.15, 0.15, 0.15, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
        
        #Rotate camera
        """
        self.camera.theta = self.angles[self.frameNum, 0]
        self.camera.phi = self.angles[self.frameNum, 1]
        self.camera.updateVecsFromPolar()
        """

        #Set up modelview matrix
        self.camera.gotoCameraFrame()
        
        #Update position and color buffers for this point
        self.mesh.needsDisplayUpdate = True
        #glLightfv(GL_LIGHT0, GL_POSITION, np.array([0, 0, 0, 1]))
        glDisable(GL_LIGHTING)
        self.mesh.renderGL(False, False, True, False, False, False, False)
        #Draw arrows
        glLineWidth(4)
        glBegin(GL_LINES)
        for i in range(self.XTraj.shape[0]-2):
            X = np.array(self.XTraj[i:i+3, :])
            dX = X[1, :] - X[0, :]
            dX = 0.5*dX
            r = np.sqrt(np.sum(dX**2))
            X1 = X[0, :] + dX
            if i < self.C.shape[0]:
                C = self.C[i, :]
            else:
                C = self.C[-1, :]
            glColor3f(C[0], C[1], C[2])
            glVertex3f(X[0, 0], X[0, 1], X[0, 2])
            glVertex3f(X1[0], X1[1], X1[2])
            #Draw arrow head
            dX2 = X[2, :] - X[1, :]
            dX = dX/np.sqrt(np.sum(dX**2))
            dX2 = dX2 - dX*np.sum(dX*dX2)
            dX2 = dX2/np.sqrt(np.sum(dX2**2))
            Y = X[0, :] + 0.8*r*dX + r*0.2*dX2
            glVertex3f(X1[0], X1[1], X1[2])
            glVertex3f(Y[0], Y[1], Y[2])
            Y = X[0, :] + 0.8*r*dX - r*0.2*dX2
            glVertex3f(X1[0], X1[1], X1[2])
            glVertex3f(Y[0], Y[1], Y[2])
        glEnd()

        self.mesh2.renderGL(False, False, True, False, False, False, False)

        saveImageGL(self, "View1.png")
        """
        if self.frameNum < self.Y.shape[0] - 1:
            self.frameNum += 1
            self.Refresh()
        else:
            self.parent.Destroy()
        
        """

        self.SwapBuffers()
#Test with a helix
if __name__ == '__main__':
    res = sio.loadmat("SphereData.mat")
    u, XTraj, C = res["u"], res["XTraj"], res["C"]

    app = wx.PySimpleApp()
    frame = wx.Frame(None, wx.ID_ANY, "PCA GL Canvas", DEFAULT_POS, (1200, 1200))
    g = SphereGLCanvas(frame, u, XTraj, C)
    frame.canvas = g
    frame.Show()
    app.MainLoop()
    app.Destroy()
