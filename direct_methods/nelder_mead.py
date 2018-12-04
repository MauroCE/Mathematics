import numpy as np


def f(x, y):
    """
    Function to be optimized. In this case this is a function
    f:R2 --> R

    :param x: First coordinate
    :type x: float
    :param y: second coordinate
    :type y: float
    """
    return 2 * x ** 2 - x * y + y ** 2 - 7 * x - y


def convert(iterable):
    """
    Converts iterable into numpy array with dtype float. This is so that
    it can be much quicker to use these functions. For instance, we can just
    use tuples.
    """
    return np.array(iterable, dtype=np.float)


def centroid(s, l):
    """
    Calculates the centroid between x_s and x_l. This will be use for the
    reflection.
    """
    s = convert(s)
    l = convert(l)
    return 0.5 * (s + l)


def reflect(c, h, alpha=1.0):
    """
    Reflects x_h with respect to x_c using alpha as stretching parameter.
    """
    c = convert(c)
    h = convert(h)
    return c + alpha * (c - h)


def expand(c, o, gamma=2.0):
    """
    It expands in direction of x_o when this is an improvement on x_l.
    """
    return c + gamma * (o - c)


def contract_forward(c, o, beta=0.5):
    """
    Contracts the new point in case x_o is less than x_s but still better than
    x_h. The contraction is still forward wrt x_c
    """
    c = convert(c)
    o = convert(o)
    return c + beta * (o - c)


def contract_backward(c, h, beta=0.5):
    """
    Similar to contract_forward, but happens when worse than x_h. Contracts
    backwards wrt x_c
    """
    return c + beta * (h - c)


def reorder(one, two, three, f=f):
    """
    Reorders the points and returns them in the order
    highest, second highest, lowest
    f is the function that we use to order
    """
    points = [one, two, three]
    return sorted(points, key=lambda x: f(*x), reverse=True)


def display(h, s, l, fh, fs, fl):
    """
    Quick function that displays the current points and their values
    """
    print("x_h = {h}   f(x_h) = {fh}".format(h=h, fh=fh))
    print("x_s = {s}   f(x_s) = {fs}".format(s=s, fs=fs))
    print("x_l = {l}   f(x_l) = {fl}".format(l=l, fl=fl))


def half_distances(l, s, h):
    """
    This function is to be applied when there is no improvement not even with
    contraction. Keep only l the same, s and h get distances from l halved.
    """
    s = l + 0.5 * (s - l)
    h = l + 0.5 * (h - l)
    return h, s, l


def nelder_mead(a, b, c, f, max_iter=4):
    """
    Carries out max_iter iterations of nelder mead algorithm.
    """
    # Store initial points and their values
    names = ["A", "B", "C"]
    initial_points = [convert(a), convert(b), convert(c)]
    initial_values = [f(*point) for point in initial_points]
    print("#" * 30 + " Initial Points " + "#" * 40)
    for n, p, v in zip(names, initial_points, initial_values):
        print("{name} = {point}      f({point}) = {value}".format(
            name=n, point=p, value=v))
    # Label the points based on their values
    print("-" * 40 + " Renaming points based on function values " + "-" * 40)
    h = initial_points.pop(np.argmax(initial_values))  # highest value
    l = initial_points.pop(np.argmin(initial_values))  # lowest value
    s = initial_points.pop()  # second highest value
    fh = initial_values.pop(np.argmax(initial_values))
    fl = initial_values.pop(np.argmin(initial_values))
    fs = initial_values.pop()
    print("x_h = {h}   f(x_h) = {fh}".format(h=h, fh=fh))
    print("x_s = {s}   f(x_s) = {fs}".format(s=s, fs=fs))
    print("x_l = {l}   f(x_l) = {fl}".format(l=l, fl=fl))
    # Set number of iterations to 1
    iteration = 1
    # keep doing the algorithm until max_iterations achieved
    while iteration <= max_iter:
        # Display iteration number and current points
        print("\n" + "#" * 40 + " iteration {it} ".format(
            it=iteration) + "#" * 40)
        print("-" * 40 + " Current Points " + "-" * 40)
        display(h, s, l, fh, fs, fl)

        # calculate centroid
        print("-" * 40 + " Calculations " + "-" * 40)
        c = centroid(s, l)
        print("Calculate Centroid: x_c = {c}".format(c=c))
        # do a reflection
        o = reflect(c, h)
        fo = f(*o)
        print("Calculate Reflection: x_o = {o}    f(x_o) = {fo}".format(o=o,
                                                                        fo=fo))
        # check the value of the reflection
        if fl <= fo <= fs:
            # replace xh by xo
            print(
                "Replace x_h by x_o: f(x_l) = {fl} <= {fo} = f(x_o)"
                " <= {fs} = f(x_s)".format(
                    fl=fl, fo=fo, fs=fs))
            h, s, l = reorder(l, s, o)
            fh, fs, fl = f(*h), f(*s), f(*l)
        elif fo < fl:
            # expand
            print("Expand: f(x_o) = {fo} < {fl} = f(x_l)".format(fo=fo, fl=fl))
            oo = expand(c, o)
            foo = f(*oo)
            print("Expanded point x_00: {}".format(oo, foo))
            if foo < fl:
                print(
                    "Replace x_h by x_oo: f(x_oo) = {foo} < {fl} = f(x_l)"
                    .format(foo=foo, fl=fl)
                )
                # replace x_h by x_oo
                h, s, l = reorder(l, s, oo)
                fh, fs, fl = f(*h), f(*s), f(*l)
            elif foo >= fl:
                print(
                    "Replace x_h by x_o: f(x_l) = {fl} <= {foo} = f(x_oo)"
                    .format(fl=fl, foo=foo))
                # replace x_h by x_o
                h, s, l = reorder(l, s, o)
                fh, fs, fl = f(*h), f(*s), f(*l)
        elif fs < fo:
            # contraction
            if fo < fh:
                # contract forward
                print(
                    "Contract forward: f(x_s) = {fs} < {fo} = f(x_o)"
                    " < {fh} = f(x_h)".format(
                        fs=fs, fo=fo, fh=fh))
                oo = contract_forward(c, o)
                foo = f(*oo)
            elif fh <= fo:
                # contract backwards
                print(
                    "Contract Backward: f(x_h) = {fh} <= {fo} = f(x_o)".format(
                        fo=fo, fh=fh))
                oo = contract_backward(c, h)
                print("Contracted Point: {}".format(oo))
                foo = f(*oo)
            # Now that we have f_oo we need to check what to do with it
            if foo < fh and foo < fo:
                # replace x_h by x_oo
                print(
                    "Replace x_h by x_oo: f(x_oo) = {foo} < {fh} = f(x_h)"
                    "   AND   f(x_oo) = {foo} < {fo} = f(x_o)".format(
                        fo=fo, foo=foo, fh=fh))
                h, s, l = reorder(l, s, oo)
                fh, fs, fl = f(*h), f(*s), f(*l)
            elif fh <= foo or fo < foo:
                # half the distances from l
                print(
                    "Half distances: f(x_h) = {fh} <= {foo} = f(x_oo)"
                    "    OR    f(x_o) = {fo} < {foo} = f(x_oo)".format(
                        fh=fh, fo=fo, foo=foo))
                h, s, l = half_distances(l, s, h)
                fh, fs, fl = f(*h), f(*s), f(*l)

        iteration += 1
    print("<" * 40 + " Final Points " + ">" * 40)
    display(h, s, l, fh, fs, fl)


if __name__ == "__main__":
    nelder_mead((0, 0), (-1, 0), (0, -1), f)
