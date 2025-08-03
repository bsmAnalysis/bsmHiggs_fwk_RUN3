import numpy as np
import awkward as ak
import os
import argparse
import vector
vector.register_awkward()
from coffea import processor
from coffea.analysis_tools import Weights
import coffea.util
import itertools

def delta_phi_raw(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi
    
def delta_phi(obj1, obj2):
    dphi = obj1.phi - obj2.phi
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def delta_eta(obj1, obj2):
    return obj1.eta - obj2.eta

def delta_r(obj1, obj2):
    deta = obj1.eta - obj2.eta
    dphi = np.abs(obj1.phi - obj2.phi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)

def clean_by_dr(objects, others, drmin):
    pairs = ak.cartesian([objects, others], nested=True)
    dr = delta_r(pairs['0'], pairs['1'])
    mask = ak.all(dr > drmin, axis=-1)
    return objects[mask]

