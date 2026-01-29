def duffy_map_corner(corner, u, v):

    if corner == 0:
        xi1 = u * (1 - v)
        xi2 = u * v
        J = u
    elif corner == 1:
        xi1 = 1.0 - u
        xi2 = u * v
        J = u
    else:  # corner == 2
        xi1 = u * v
        xi2 = 1.0 - u
        J = u
    return xi1, xi2, J
