from dolfin import *
from ufl import tanh
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EnumUtilities import BrainSection

def get_initial_condition(plane=None):

    R0   = Constant(2.0)
    beta = Constant(1.0)

    if plane == BrainSection.SAGITTAL:
        xc, yc = -16.0, 3.5
        radius = lambda x: sqrt(pow((x[0] - xc), 2) + pow(1*(x[1] - yc), 2))
        c_0    = lambda x, t: 0.25*(1.0 + tanh(4.0 - beta*(radius(x) - R0)))**2
        return c_0
    elif plane == BrainSection.CORONAL:
        xc, yc = 8.0, -0.02
        radius = lambda x: sqrt(pow((x[0] - xc), 2) + pow(5*(x[1] - yc), 2))
        c_0    = lambda x, t: 0.25*(1.0 + tanh(4.0 - beta*(radius(x) - R0)))**2
        return c_0
    elif plane == BrainSection.HORIZONTAL:
        xc, yc = 0.0, 0.0
        radius = lambda x: sqrt(pow((x[0] - xc), 2) + pow((x[1] - yc), 2))
        c_0    = lambda x, t: 0.25*(1.0 + tanh(4.0 - beta*(radius(x) - R0)))**2
        return c_0
    else:
        AssertionError("No plane provided")
        sys.exit(0)