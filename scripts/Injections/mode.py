from dataclasses import dataclass

@dataclass(frozen = True, eq = True)
class Mode:
    p : int = -1
    s : int = -2
    l : int = 2
    m : int = 2
    n : int = 0
    
    def to_tuple(self):
        return (self.p,self.s,self.l,self.m,self.n)
    
    @property
    def inputs(self):
        return {'p': self.p, 's': self.s, 'l': self.l, 'm': self.m, 'n': self.n}