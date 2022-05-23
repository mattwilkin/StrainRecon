import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d
import util_mjw.Simulation as Gsim
import util_mjw.RotRep as Rot
from scipy import ndimage
from scipy import optimize
from scipy.ndimage import center_of_mass
from scipy.ndimage.measurements import label,find_objects
import json
import h5py
import yaml
from scipy.interpolate import griddata
from util_mjw.MicFileTool import MicFile
from util_mjw.config import Config
import shutil
import os

# extract window around the Bragg peak on an omega frame
def fetch(ii,pks,fn,offset=0,dx=100,dy=50,verbo=False,more=False,pnx=2048,pny=2048,omega_step=20):
    omegid=int((180-pks[ii,2])*omega_step)+offset
    if omegid<0:
        omegid+=3600
    if omegid>=3600:
        omegid-=3600
    I=plt.imread(fn+'{0:06d}.tif'.format(omegid))
    x1=int((pny-1-pks[ii,0])-dx)
    y1=int(pks[ii,1]-dy)
    if verbo:
        print('y=',pks[ii,1])
        print('x=',pks[ii,0])
    x1=max(0,x1)
    y1=max(0,y1)
    x2=x1+2*dx
    y2=y1+2*dy
    x2=min(x2,pnx)
    y2=min(y2,pny)
    if more:
        return I[y1:y2,x1:x2],(x1,x2,y1,y2,omegid)
    return I[y1:y2,x1:x2]
# choose one of following two methods to find the center of mass of each "good" Bragg peak.




def getCenter2(Im,Omeg,dx=15,dy=7,do=2):
    Py,Px=ndimage.measurements.maximum_position(Im[Omeg])
    labels=np.zeros(Im.shape,dtype=int)
    x_window = np.array([Px-dx,Px+dy])
    y_window = np.array([Py-dy,Py+dy])
    x_window[x_window<0] = 0
    x_window[x_window>Im.shape[2]] = Im.shape[2]
    y_window[y_window<0] = 0
    y_window[y_window>Im.shape[1]] = Im.shape[1]
    o_window = np.array([Omeg-do,Omeg+do])
    o_window[o_window<0] = 0
    o_window[o_window>Im.shape[0]] = Im.shape[0]
    
    # print(o_window,labels.shape)
    labels[o_window[0]:o_window[1],y_window[0]:y_window[1],x_window[0]:x_window[1]]=1
    
    co,cy,cx = center_of_mass(Im,labels=labels,index=1)
    return Py,Px,cy,cx,co


def fetch_images(Cfg,grain,path):
    
    if f'grain_%03d'%grain.grainID in os.listdir(path):
        shutil.rmtree(path+'grain_%03d/'%grain.grainID)
        
    os.mkdir(path+'grain_%03d/'%grain.grainID)
    os.mkdir(path+'grain_%03d/RawImgData/'%grain.grainID)
    os.mkdir(path+'grain_%03d/FilteredImgData/'%grain.grainID)
    
    Det1=Gsim.Detector(config=Cfg)
    crystal_str=Gsim.CrystalStr(config=Cfg)
    crystal_str.getRecipVec()
    crystal_str.getGs(Cfg.maxQ)
    o_mat=Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
    pks,Gs,Info=Gsim.GetProjectedVertex(Det1,crystal_str,o_mat,Cfg.etalimit/180*np.pi,
                                        grain.grainPos,getPeaksInfo=True,
                                        omegaL=Cfg.omgRange[0],omegaU=Cfg.omgRange[1],energy=Cfg.energy)
    
    dx,dy= Cfg.dx,Cfg.dy
    window = Cfg.window
    raw_data = Cfg.dataFile
    
    rng_low = window[2]//2
    if window[2]%2 == 0:
        rng_high = rng_low
    else:
        rng_high = rng_low + 1
    for ii in range(len(pks)):
        allpks=[]
        alllims=[]
        totoffset=0
        # f,axis=plt.subplots(9,5)

        for offset in range(totoffset-rng_low,totoffset+rng_high):
            Im,limits=fetch(ii,pks,raw_data,offset,dx=dx,dy=dy,more=True)


            # ax.imshow(Im,vmin=0,vmax=30)        
            allpks.append(Im)
            alllims.append(limits)

        # f.subplots_adjust(wspace=0,hspace=0)
        # f.savefig(path+'Ps_bf/%03d.png'%ii,dpi=200,bbox_inches='tight')
        # plt.close(f)
        allpks=np.array(allpks)
        alllims=np.array(alllims)
        np.save(path+'/grain_%03d/RawImgData/Im_%03d'%(grain.grainID,ii),allpks)
        np.save(path+'/grain_%03d/RawImgData/limit_%03d'%(grain.grainID,ii),alllims)
    return 
def process_images(grain,path,window,flucThresh):
    Nfile = len([f for f in os.listdir(path+'grain_%03d/RawImgData'%grain.grainID) if f[:2] == 'Im'])
    Im=[]
    print('thresholding')
    for ii in range(Nfile):
        Im.append(np.load(path+'grain_%03d/RawImgData/Im_%03d.npy'%(grain.grainID,ii)))
        Im[ii]=Im[ii]-np.median(Im[ii],axis=0) #substract the median
        mask=Im[ii]>flucThresh
        Im[ii]=mask*Im[ii] #make all pixel that below the fluctuation to be zero 

    print('removing hot spots')
    mykernel=np.array([[1,1,1],[1,-1,1],[1,1,1]])
    # remove hot spot (whose value is higher than the sum of 8 neighbors)
    for ii in range(Nfile):
        for jj in range(window[2]):
            mask=convolve2d(Im[ii][jj],mykernel,mode='same')>0
            Im[ii][jj]*=mask

    print('smoothing')
    mykernel2=np.array([[1,2,1],[2,4,2],[1,2,1]])/16.0
    # Smoothing
    for ii in range(Nfile):
        for jj in range(window[2]):
            Im[ii][jj]=convolve2d(Im[ii][jj],mykernel2,mode='same')

    for ii in range(Nfile):
        np.save(path+'grain_%03d/FilteredImgData/Im_%03d.npy'%(grain.grainID,ii),Im[ii].astype('uint16'))
    return
def GetVertex(Det1,Gs,Omegas,orien,etalimit,grainpos,bIdx=True,omegaL=-90,omegaU=90,energy=50):
    Peaks=[]
    rotatedG=orien.dot(Gs.T).T
    for ii in range(len(rotatedG)):
        g1=rotatedG[ii]
        res=Gsim.frankie_angles_from_g(g1,verbo=False,energy=energy)

        if Omegas[ii]==1:
            omega=res['omega_a']/180.0*np.pi
            newgrainx=np.cos(omega)*grainpos[0]-np.sin(omega)*grainpos[1]
            newgrainy=np.cos(omega)*grainpos[1]+np.sin(omega)*grainpos[0]
            idx=Det1.IntersectionIdx(np.array([newgrainx,newgrainy,0]),res['2Theta'],res['eta'],bIdx
                                     ,checkBoundary=False
                                    )

            Peaks.append([idx[0],idx[1],res['omega_a']])

                
        else:
            omega=res['omega_b']/180.0*np.pi
            newgrainx=np.cos(omega)*grainpos[0]-np.sin(omega)*grainpos[1]
            newgrainy=np.cos(omega)*grainpos[1]+np.sin(omega)*grainpos[0]
            idx=Det1.IntersectionIdx(np.array([newgrainx,newgrainy,0]),res['2Theta'],-res['eta'],bIdx
                                    ,checkBoundary=False
                                    )
            Peaks.append([idx[0],idx[1],res['omega_b']])

    Peaks=np.array(Peaks)
    return Peaks

def optimize_detector(center_of_mass,goodidx,grain,cutoff=[ 60,30,10]):
    
    Cfg = Config(grain.configFile)
    Det1=Gsim.Detector(config=Cfg)

    crystal_str=Gsim.CrystalStr(config=Cfg)

    crystal_str.getRecipVec()
    crystal_str.getGs(Cfg.maxQ)


  
    o_mat=Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
    pks,Gs,Info=Gsim.GetProjectedVertex(Det1,crystal_str,o_mat,Cfg.etalimit/180*np.pi,
                                        np.array(grain.grainPos),getPeaksInfo=True,
                                        omegaL=Cfg.omgRange[0],omegaU=Cfg.omgRange[1],energy=Cfg.energy)
    
    
    
    pars={'J':0,'K':0,'L':0,'tilt':(0,0,0),'x':0,'y':0,'distortion':((0,0,0),(0,0,0),(0,0,0))}
    DetDefault=Gsim.Detector(psizeJ=Cfg.pixelSize*1e-3, psizeK=Cfg.pixelSize*1e-3)
    
    def SimP(x):

        DetDefault.Reset()
        pars['J']=x[0]+ Cfg.JCenter 
        pars['K']=x[1]+ Cfg.KCenter 
        pars['L']=x[2]*10**(-3) + Cfg.Ldistance 
        pars['tilt']= Rot.EulerZXZ2Mat((x[3:6]+np.array(Cfg.tilt))/180*np.pi)
        pars['x']=x[6]*10**(-3) + grain.grainPos[0]
        pars['y']=x[7]*10**(-3) + grain.grainPos[1]
        pars['distortion']=x[8:17].reshape((3,3))*10**(-3)+np.eye(3)
        DetDefault.Move(pars['J'],pars['K'],np.array([pars['L'],0,0]),pars['tilt'])
        pos=np.array([pars['x'], pars['y'], 0])
        Ps=GetVertex(DetDefault,
                        good_Gs,
                        whichOmega,
                        pars['distortion'],
                        Cfg.etalimit/180*np.pi,
                        pos,
                        bIdx=False,
                        omegaL=Cfg.omgRange[0],omegaU=Cfg.omgRange[1],energy=Cfg.energy) 
        
        return Ps


    def CostFunc(x):
        Ps = SimP(x)
        weights=np.array((1,5,100))
        tmp=np.sum(((Ps-absCOM)*weights)**2,axis=0)
        return np.sum(tmp)
    
    
    tolerance = cutoff
    tolerance.append(tolerance[-1])
    for tol in tolerance:
        imgN = len(goodidx)

        LimH = np.empty((imgN,5),dtype=np.int32)
        good_Gs = Gs[goodidx]
        whichOmega = np.empty(imgN,dtype=np.int32)


        for ii in range(imgN):
            limit=np.load('Calibration_Files/grain_%03d/RawImgData/limit_%03d.npy'%(grain.grainID,goodidx[ii]))
            LimH[ii,:]=limit[0]

            if Info[goodidx[ii]]['WhichOmega']=='b':
                whichOmega[ii] = 2
            else:
                whichOmega[ii] = 1

        absCOM=np.empty(center_of_mass.shape)
        for ii in range(len(absCOM)):
            absCOM[ii,1]=LimH[ii,2]+center_of_mass[ii,2]
            absCOM[ii,0]=Cfg.JPixelNum-1-(LimH[ii,0]+center_of_mass[ii,1])
            absCOM[ii,2]=(LimH[ii,4]+center_of_mass[ii,0])
            if absCOM[ii,2] >= 3600:
                absCOM[ii,2] -= 3600
            absCOM[ii,2] = 180-absCOM[ii,2]*Cfg.omgInterval







        res=optimize.minimize(CostFunc,np.zeros(17)
                              ,bounds=[(-5,5),(-5,2),(-100,50)]+3*[(-0.3,3)]+2*[(-10,20)]+9*[(-5,10)]
                             )
        newPs=SimP(res['x'])
        oldPs=SimP(np.zeros(17))
        dists = np.absolute(np.linalg.norm(newPs-absCOM,axis=1))
        inds = np.where(dists<tol)
        goodidx = goodidx[inds]
        center_of_mass = center_of_mass[inds]
        print(np.linalg.det(res['x'][8:].reshape((3,3))))
        
    return res['x'],oldPs,newPs,absCOM,goodidx

def find_grains(Cfg,conf_tol):
    a=MicFile(Cfg.hexomapFile)
    grid_x,grid_y=np.meshgrid(np.arange(-0.5,0.2,0.002),np.arange(-0.4,0.4,0.002))
    grid_c = griddata(a.snp[:,0:2],a.snp[:,9],(grid_x,grid_y),method='nearest')
    grid_e1 = griddata(a.snp[:,0:2],a.snp[:,6],(grid_x,grid_y),method='nearest')
    grid_e2 = griddata(a.snp[:,0:2],a.snp[:,7],(grid_x,grid_y),method='nearest')
    grid_e3 = griddata(a.snp[:,0:2],a.snp[:,8],(grid_x,grid_y),method='nearest')

    g = np.where(grid_c>conf_tol,1,0)

    labels,num_features = label(g)

    ll = np.float32(labels.copy())
    ll[ll==0] = np.nan


    GrainDict={}
    for l in np.sort(np.unique(labels))[1:]:

        com =center_of_mass(g,labels,l)
        com = (int(com[0]),int(com[1]))
        GrainDict[l] = (grid_e1[com],grid_e2[com],grid_e3[com])
    GrainIDMap=np.zeros(grid_c.shape,dtype=int)
    for grainID in GrainDict:

        (e1,e2,e3)=GrainDict[grainID]
        tmp = grid_c > 0.3
        tmp*=np.absolute(grid_e1 - e1)<1
        tmp*=np.absolute(grid_e2 - e2)<1
        tmp*=np.absolute(grid_e3 - e3)<1
        GrainIDMap[tmp]= grainID 
    newGrainIDMap = GrainIDMap.copy()
    for i,ggg in enumerate(np.unique(GrainIDMap)):
        newGrainIDMap[newGrainIDMap==ggg] = i
        
    GrainIDMap = newGrainIDMap

    with h5py.File(Cfg.micFile,'w') as f:
        ds=f.create_dataset("origin", data = np.array([-0.5,-0.4]))
        ds.attrs[u'units'] = u'mm'
        ds=f.create_dataset("stepSize", data = np.array([0.002,0.002]))
        ds.attrs[u'units'] = u'mm'
        f.create_dataset("Xcoordinate", data = grid_x)
        f.create_dataset("Ycoordinate", data = grid_y)
        f.create_dataset("Confidence", data = grid_c)
        f.create_dataset("Ph1", data = grid_e1)
        f.create_dataset("Psi", data = grid_e2)
        f.create_dataset("Ph2", data = grid_e3)
        f.create_dataset("GrainID", data = GrainIDMap)

    gg = np.float32(GrainIDMap.copy())
    gg[gg==0] = np.nan
    fig,ax = plt.subplots(ncols=4,figsize=(20,7))
    ax[0].imshow(grid_e3,origin='lower')
    ax[1].imshow(g,origin='lower')
    ax[2].imshow(ll,origin='lower')
    ax[3].imshow(gg,origin='lower')
    plt.show()
    print('Number of Grains:',GrainIDMap.max())
    
    
    grain_Ids = np.unique(GrainIDMap)[1:]
    
    

    grain_posi = []

    for i in grain_Ids:
        with open(Cfg.grainTemp) as f:
            data = yaml.safe_load(f)
        i = int(i)
        locations = np.where(GrainIDMap==i,1,0)

        com_ind = np.int32(np.round(center_of_mass(locations)))

        grain_pos = np.round(np.array([ grid_x[com_ind[0],com_ind[1]],grid_y[com_ind[0],com_ind[1]],0]),4)
        grain_posi.append(grain_pos)
        euler = np.array([grid_e1[locations==1].mean(),grid_e2[locations==1].mean(),grid_e3[locations==1].mean()])


        data['grainID'] = i
        data['peakFile'] = f'Peak_Files/Peaks_g{i}.hdf5'
        data['recFile'] = f'Rec_Files/Rec_g{i}.hdf5'
        data['grainPos'] = [float(g) for g in grain_pos]
        data['euler'] = [float(e) for e in euler]

        with open(f'Config_Files/Grain_Files/Grain_%03d.yml'%i, 'w') as file:

            documents = yaml.dump(data, file)
    
    
    
    
    return GrainIDMap.max()

def data_prep(Cfg,grain,path,flucThresh=4):
    
    fetch_images(Cfg,grain,path)
    

    process_images(grain,path,Cfg.window,flucThresh)
    
    
    
    
    center_of_mass = []
    Nfile = len([f for f in os.listdir(path+'grain_%03d/RawImgData'%grain.grainID) if f[:2] == 'Im'])
    goodidx = np.arange(Nfile)

    for idx in goodidx:
        tmp=np.load(path+'grain_%03d/FilteredImgData/Im_%03d.npy'%(grain.grainID,idx))
        Omeg = np.argmax(tmp[Cfg.window[2]//3:2*Cfg.window[2]//3,
                             Cfg.window[1]//3:2*Cfg.window[1]//3,
                             Cfg.window[0]//3:2*Cfg.window[0]//3].sum(axis=(1,2)))+Cfg.window[2]//3
        Py,Px,cy,cx,co = getCenter2(tmp,Omeg,dx=50,dy=50,do=5)
        center = np.array([co,cx,cy])
        center_of_mass.append(center)
    center_of_mass = np.stack(center_of_mass)
    return center_of_mass,goodidx

def write_config_files(x,Num_Grains):
    


    with open(f'Config_Files/Config.yml') as f:
        data = yaml.safe_load(f)


    with open('Config_Files/Config.yml', 'w') as file:
        data['JCenter'] += float(x[0])
        data['KCenter'] += float(x[1])
        data['Ldistance'] += float(x[2]*1e-3)
        data['tilt'] = [float(a) for a in list(np.array(data['tilt'])+x[3:6])]
        documents = yaml.dump(data, file)
    for g in range(1,Num_Grains+1):
        with open('Config_Files/Grain_Files/Grain_%03d.yml'%g, 'r') as file:
            data = yaml.safe_load(file)
        
            
        data['grainPos'] = [float(a) for a in list(np.array(data['grainPos'])+np.array([x[6],x[7],0])*1e-3)]
        data['peakFile'] = 'Peak_Files/grain_%03d/Peaks_g%03d.hdf5'%(g,g)
        data['RecFile'] = 'Rec_Files/Rec_g%03d.hdf5'%g
        with open('Config_Files/Grain_Files/Grain_%03d.yml'%g, 'w') as file:
            documents = yaml.dump(data, file)
    return

def optimize_distortion(Cfg,grain,center_of_mass,path):
    
    Cfg = Config(grain.configFile)
    Det1=Gsim.Detector(config=Cfg)

    crystal_str=Gsim.CrystalStr(config=Cfg)

    crystal_str.getRecipVec()
    crystal_str.getGs(Cfg.maxQ)


  
    o_mat=Rot.EulerZXZ2Mat(np.array(grain.euler)/180.0*np.pi)
    pks,Gs,Info=Gsim.GetProjectedVertex(Det1,crystal_str,o_mat,Cfg.etalimit/180*np.pi,
                                        np.array(grain.grainPos),getPeaksInfo=True,
                                        omegaL=Cfg.omgRange[0],omegaU=Cfg.omgRange[1],energy=Cfg.energy)
    
    
    imgN = len(pks)

    LimH = np.empty((imgN,5),dtype=np.int32)
    good_Gs = Gs
    whichOmega = np.empty(imgN,dtype=np.int32)


    for ii in range(imgN):
        limit=np.load(path+'/grain_%03d/RawImgData/limit_%03d.npy'%(grain.grainID,ii))
        LimH[ii,:]=limit[0]

        if Info[ii]['WhichOmega']=='b':
            whichOmega[ii] = 2
        else:
            whichOmega[ii] = 1

    absCOM=np.empty(center_of_mass.shape)
    for ii in range(len(absCOM)):
        absCOM[ii,1]=LimH[ii,2]+center_of_mass[ii,2]
        absCOM[ii,0]=Cfg.JPixelNum-1-(LimH[ii,0]+center_of_mass[ii,1])
        absCOM[ii,2]=(LimH[ii,4]+center_of_mass[ii,0])
        if absCOM[ii,2] >= 3600:
            absCOM[ii,2] -= 3600
        absCOM[ii,2] = 180-absCOM[ii,2]*Cfg.omgInterval
    pars={'J':0,'K':0,'L':0,'tilt':(0,0,0),'x':0,'y':0,'distortion':((0,0,0),(0,0,0),(0,0,0))}
    DetDefault=Gsim.Detector(psizeJ=Cfg.pixelSize*1e-3, psizeK=Cfg.pixelSize*1e-3)
    
    def SimP(x):

        DetDefault.Reset()
        pars['J']=Cfg.JCenter 
        pars['K']=Cfg.KCenter 
        pars['L']= Cfg.Ldistance 
        pars['tilt']= Rot.EulerZXZ2Mat(np.array(Cfg.tilt)/180*np.pi)
        pars['x']=grain.grainPos[0]
        pars['y']=grain.grainPos[1]
        pars['distortion']=x.reshape((3,3))*10**(-3)+np.eye(3)
        DetDefault.Move(pars['J'],pars['K'],np.array([pars['L'],0,0]),pars['tilt'])
        pos=np.array([pars['x'], pars['y'], 0])
        Ps=GetVertex(DetDefault,
                        good_Gs,
                        whichOmega,
                        pars['distortion'],
                        Cfg.etalimit/180*np.pi,
                        pos,
                        bIdx=False,
                        omegaL=Cfg.omgRange[0],omegaU=Cfg.omgRange[1],energy=Cfg.energy) 
        
        return Ps
    def CostFunc(x):
        Ps = SimP(x)
        weights=np.array((1,5,100))
        tmp=np.sum(((Ps-absCOM)*weights)**2,axis=0)
        return np.sum(tmp)
    
    res=optimize.minimize(CostFunc,np.zeros(9)
                              ,bounds=9*[(-5,10)]
                             )
    newPs=SimP(res['x'])
    oldPs=SimP(np.zeros(9))
    dists = np.absolute(np.linalg.norm(newPs-absCOM,axis=1))
    
    return res['x'],newPs,absCOM,good_Gs,Info