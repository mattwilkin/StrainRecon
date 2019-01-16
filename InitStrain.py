import util.Simulation as Gsim
import util.RotRep as Rot
import numpy as np
import matplotlib.pyplot as plt
import yaml

class Initializer(object):

    def __init__(self,cfgFn='ConfigFiles/g15Ps1_2nd.yml'):

        with open(cfgFn) as f:
            dataMap=yaml.safe_load(f)

        ##############
        # Files
        ##############
        self.peakfn=dataMap['Files']['peakFile']
        self.micfn=dataMap['Files']['micFile']



        self.exp={'energy':dataMap['Setup']['Energy']}
        self.eng=self.exp['energy']
        self.etalimit=dataMap['Setup']['Eta-Limit']/180.0*np.pi
        self.omgRange=dataMap['Setup']['Omega-Range'] 
        ########################
        # Detector parameters
        ########################

        self.Det=Gsim.Detector(psizeJ=dataMap['Setup']['Pixel-Size']/1000.0,
                psizeK=dataMap['Setup']['Pixel-Size']/1000.0,
                J=dataMap['Setup']['J-Center'],
                K=dataMap['Setup']['K-Center'],
                trans=np.array([dataMap['Setup']['Distance'],0,0]),
                tilt=Rot.EulerZXZ2Mat(np.array(dataMap['Setup']['Tilt'])/180.0*np.pi))

        #########################
        # LP
        #########################
        self.Ti7LP=Gsim.CrystalStr()
        self.Ti7LP.PrimA=dataMap['Material']['Lattice'][0]*np.array([1,0,0])
        self.Ti7LP.PrimB=dataMap['Material']['Lattice'][1]*np.array([np.cos(np.pi*2/3),np.sin(np.pi*2/3),0])
        self.Ti7LP.PrimC=dataMap['Material']['Lattice'][2]*np.array([0,0,1])
        Atoms=dataMap['Material']['Atoms']
        for ii in range(len(Atoms)):
            self.Ti7LP.addAtom(list(map(eval,Atoms[ii][0:3])),Atoms[ii][3])
        self.Ti7LP.getRecipVec()
        self.Ti7LP.getGs(dataMap['Material']['MaxQ'])



    def Simulate(self):
        self.Ps,self.Gs,self.Info=Gsim.GetProjectedVertex(self.Det,
                self.Ti7LP,self.orienM,self.etalimit,
                self.pos,getPeaksInfo=True,
                omegaL=self.omgRange[0],
                omegaU=self.omgRange[1],**(self.exp))
        self.NumG=len(self.Gs)
    
    def SetPosOrien(self,pos,orien):
        self.pos=np.array(pos)
        self.orien=np.array(orien)
        self.orienM=Rot.EulerZXZ2Mat(self.orien/180.0*np.pi)
