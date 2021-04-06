#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import utilitiesForTests as tUtil

def computePads(mu,npads_cath0,npads_cath1):
    x0, y0, dx0, dy0 = tUtil.buildPads( npads_cath0[0], npads_cath0[1], -1.0, 1.0, -1.0, 1.0 )
    x1, y1, dx1, dy1 = tUtil.buildPads( npads_cath1[0], npads_cath1[1], -1.0, 1.0, -1.0, 1.0 )
    
    # Merge pads
    (x, y, dx, dy) = tUtil.mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )

    N0 = x0.size
    cath = np.zeros( x.size, dtype=np.int32 )
    cath[N0:] = 1
    return (x,y,dx,dy,cath)

def computeMathieson(mu,npads_cath0,npads_cath1):
    chId = 2
    
    (x,y,dx,dy,cath) = computePads(mu,npads_cath0,npads_cath1)
    #
    # xyInfSup
    xInf = x - dx - mu[0]
    xSup = x + dx - mu[0]
    yInf = y - dy - mu[1]
    ySup = y + dy - mu[1]

    z = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
    return (x,y,dx,dy,cath,z)

def makePlot(x,y,dx,dy,cath,z,title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[0], x[cath==0], y[cath==0], dx[cath==0],
    dy[cath==0], z[cath==0],  title="%s cath-0" % title)
    uPlt.drawPads( fig, ax[1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1],  
    title="%s cath-1" % title)
    plt.show()
    
def plotMathieson(*,mu,npads_cath0,npads_cath1):
    (x,y,dx,dy,cath,z) = computeMathieson(mu,npads_cath0,npads_cath1)
    makePlot(x,y,dx,dy,cath,z,"Mathieson ({},{})".format(mu[0],mu[1]))
    print("mu={},{} sum z={}".format(mu[0],mu[1], np.sum(z)))

def plotMathiesonMixture(*,mu,npads_cath0,npads_cath1):

    (x,y,dx,dy,cath) = computePads(mu,npads_cath0,npads_cath1)
    
    #
    # Mathieson mixture
    #
    chId = 2
    w   = np.array( [0.6, 0.4 ] )
    muX = np.array( [0.4, -0.4] )
    muY = np.array( [0.4, -0.4] )
    cstVar = 0.1 * 0.1
    varX = np.array( [ cstVar, cstVar] )
    varY = np.array( [ cstVar, cstVar] )
    theta = tUtil.asTheta( w, muX, muY, varX, varY)
    xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy)
    zMix = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chId )
    print("sum zMix", np.sum(zMix))
   
    makePlot(x,y,dx,dy,cath,zMix,"Mathieson")

def makePlots(npads_cath0,npads_cath1):
    plotMathieson(mu=[0.0,0.0],npads_cath0=npads_cath0,npads_cath1=npads_cath1)
    plotMathieson(mu=[0.4,0.5],npads_cath0=npads_cath0,npads_cath1=npads_cath1)
    plotMathiesonMixture(mu=[0.4,0.5],npads_cath0=npads_cath0,npads_cath1=npads_cath1)

if __name__ == "__main__":
    makePlots(npads_cath0=[4,2],npads_cath1=[2,4])
