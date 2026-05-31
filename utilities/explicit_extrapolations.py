from utilities.enum_utilities import BdfOrder

# Dictionary of lambdas for explicit extrapolations:
#    u[j] = u^{n - j}
EXPLICIT_EXTRAPOLATIONS = {
    BdfOrder.BDF1 : (lambda u: u[0]),
    BdfOrder.BDF2 : (lambda u: 2.0*u[0] - u[1]),
    BdfOrder.BDF3 : (lambda u: 3.0*u[0] - 3.0*u[1] + u[2]),
    BdfOrder.BDF4 : (lambda u: 4.0*u[0] - 6.0*u[1] + 4.0*u[2] - u[3]),
    BdfOrder.BDF5 : (lambda u: 5.0*u[0] - 10.0*u[1] + 10.0*u[2] - 5.0*u[3] + u[4]),
    BdfOrder.BDF6 : (lambda u: 6.0*u[0] - 15.0*u[1] + 20.0*u[2] - 15.0*u[3] + 6.0*u[4] - u[5]),
}