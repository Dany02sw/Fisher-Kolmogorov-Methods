from abc import ABC, abstractmethod
from dolfin import exp, ln, Constant

# Abstractclass for tranformations
class Transformation(ABC):
    @abstractmethod
    def __call__(self, w): ...
    
    @abstractmethod  
    def inv(self, c): ...

    @abstractmethod
    def invPrime(self, c): 
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the prime derivative of the inverse of the transformation"
        )

# Identity transformation for DG, LDG
class Identity(Transformation):
    def __call__(self, u):   return u
    def inv(self, c):        return c
    def invPrime(self, c):   return Constant(1.0)

# Exponential transformation for PPDG
class Exponential(Transformation):
    def __init__(self, eps=1e-10): # constructor for smoothing parameter
        self.eps = Constant(eps)

    def __call__(self, u):   return exp(u)
    def inv(self, c):        return ln(c + self.eps)
    def invPrime(self, c):   return 1/(c + self.eps)

# Entropic transformation class for SP-LDG
class EntropicTransformation(Transformation, ABC):

    def s1(self, c):    return self.inv(c)
    def s2(self, c):    return self.invPrime(c)

    @abstractmethod
    def invPrime(self, c): ...

class Sigmoid(EntropicTransformation):
    def __init__(self, eps=1e-10): # constructor for smoothing parameter
        self.eps = Constant(eps)

    def __call__(self, w):      return exp(w) / (1.0 + exp(w))
    def inv(self, c):           return ln(c + self.eps) - ln(1.0 - c + self.eps)
    def invPrime(self, c):      return 1.0 / (c * (1.0 - c) + self.eps) 