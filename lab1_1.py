import math

#machine infinity
def inf():
    inf_ = 1.
    while inf_ * 2 != math.inf:
        inf_ *= 2
    return inf_

#machine zero
def zero():
    zer0 = 1.
    while zer0 * .5 != 0.:
        zer0 *= .5
    return zer0

#machine epsilon
def eps():
    epsilon = 1.
    while 1. + epsilon * .5 != 1.:
        epsilon *= .5
    return epsilon * .5


print(f'Machine infinity: {inf():.2e}')
print(f'Machine zero: {zero():.2e}')
print(f'Machine epsilon: {eps():.2e}')