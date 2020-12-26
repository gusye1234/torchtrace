"""Another wrapper for Tracer, looking more on base.py
"""
from .base import Tracer

class tracer(Tracer):
    """
        Tracer wrapper to support bounded methods
    """
    def __init__(self, *args, **kwargs):
        super(tracer, self).__init__(*args, **kwargs)
            
    
    def from_Tracer(self, old_Tracer : Tracer):
        try:
            assert isinstance(old_Tracer, Tracer)
        except AssertionError:
            raise AssertionError(f"expect \'last\' to by Tracer type, get {type(old_Tracer)}")
        self._last_node = old_Tracer.before()
        self._is_leaf   = old_Tracer.is_leaf()
        self._required  = old_Tracer.ask_grad()
        self.data = old_Tracer.data
        return self
        
    def Gather(self, index):
        return tracer().from_Tracer(self.base_Gather(index))
        
    def View(self, *shape):
        return tracer().from_Tracer(self.base_View(*shape))
    
    def Sum(self):
        return tracer().from_Tracer(self.base_Sum())
    
    def __repr__(self):
        hints = super(Tracer, self).__repr__()
        hints = "tracer" + hints[6:]
        return hints