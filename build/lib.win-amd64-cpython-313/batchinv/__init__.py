from batchinv import _C

def rms_norm(input, weight, eps=1e-6):
    return _C.rms_norm(input, weight, eps)