import numpy as np
from copy import deepcopy


def func(point):
    """
    Function to be optimized for the coursework.
    """
    point = convert(point)
    if len(point) != 2:
        raise TypeError("This is a 2 variable function!")
    else:
        x = point[0]
        y = point[1]
        return 2 * x ** 2 - x * y + y ** 2 - 7 * x - y


def fake_func(point):
    """
    Function of the exercise given in class.
    :param point:
    :return:
    """
    point = convert(point)
    if len(point) != 2:
        raise TypeError("This is a 2 variable function!")
    else:
        x = point[0]
        y = point[1]
        return 3 * x ** 2 - 2 * x * y + y ** 2 + 4 * x + 3 * y


def convert(iterable):
    """
    Converts iterable into numpy array with dtype float. This is so that
    it can be much quicker to use these functions. For instance, we can just
    use tuples.
    """
    return np.array(iterable, dtype=np.float)


def explore(base, lengths, f):
    """
    This function does the exploratory moves for the Nelder Mead algorithm.
    The exploratory moves are done around base which is an n-dimensional
    iterable (for instance a tuple, list or array).

    :param base: base point around which to do exploratory moves.
    :type base: iterable
    :param lengths: step lengths used to explore, ordered by coordinate.
    :type lengths: iterable
    :param f: function used to evaluate the various moves
    :type f: function
    """
    # Convert to arrays
    current_point = convert(base)
    lengths = convert(lengths)
    current_value = f(current_point)
    # Iterate through the dimensions and explore
    print("-" * 40 + " Exploratory Moves " + "-" * 40)
    for dim in range(len(base)):
        # create b_1 + h_1 e_1
        move = deepcopy(current_point)
        move[dim] += lengths[dim]
        print("Exploring: {} \t Value: {}".format(move, f(move)))
        # If new move has lower value, replace current with this
        if f(move) < current_value:
            # print("Current Point: {}".format(current_point))
            # print("Move: {}".format(move))
            # print("f(move) < f(current_point) cause f(move)={} "
            #       "and f(current_point)={}".format(f(move), current_value))
            current_point = move
            current_value = f(move)

        else:
            # create b_1 - h1 e_1
            move = deepcopy(current_point)
            move[dim] -= lengths[dim]
            print("Exploring: {} \tValue: {}".format(move, f(move)))
            # if new move has lower value, replace current with this
            if f(move) < current_value:
                current_point = move
                current_value = f(move)
    print("Exploratory Result: {} \t Value: {}".format(current_point,
                                                       f(current_point)))
    return current_point


def pattern(old_point, new_point, new_point_value, f, lengths):
    """
    Pattern moves for the Hooke and Jeeves algorithm.
    Here new_point_value is f(new_point)
    """
    # calculate p_1, the "stretched" point
    p = 2 * new_point - old_point
    fp = f(p)
    print("Pattern move: {} \tValue: {}".format(p, fp))
    # explore around p
    p_explored = explore(p, lengths, f)
    fp_explored = f(p_explored)
    # Need to output the best between f(p), f(p_explored) and f(new_point)
    # If none is better than new point, return new point
    if min(fp, fp_explored) >= new_point_value:
        final_point, final_value = new_point, new_point_value
    # if p is the best, return it, otherwise return p_explored
    elif fp < fp_explored:
        final_point, final_value = p, fp
    else:
        final_point, final_value = p_explored, fp_explored
    print("Pattern+Exploratory: {} \tValue: {}"
          .format(final_point, final_value))
    return final_point, final_value


def hooke_jeeves(base, lengths, stopping_lengths, f):
    """
    Hooke & Jeeves algorithm.
    Uses a sequence of exploratory moves (implemented by the function "explore"
    ) and a sequence of pattern moves, implemented by "patter".
    """
    # Use arrays instead of other iterables
    print("#" * 40 + " Hooke and Jeeves " + "#" * 40)
    print("-" * 40 + " initial settings " + "-" * 40)
    current_point = convert(base)
    lengths = convert(lengths)
    stopping_lengths = convert(stopping_lengths)
    print("Base point: {base} \nLengths: {len} \nStopping lengths: {sl}"
          .format(base=current_point, len=lengths, sl=stopping_lengths))
    # Notice numpy uses broadcasting and element-wise comparison so we can
    # simply do lengths < stopping_lengths to check whether we should stop or ]
    # not.
    # keep track of number of iterations
    i = 1
    while np.all(lengths >= stopping_lengths):
        print("#" * 40 + " Iteration {it} ".format(it=i) + "#" * 40)
        # do exploratory moves
        new_point = explore(current_point, lengths, f)
        # If we haven't obtained an improvement, half step lengths
        if np.array_equal(new_point, current_point):
            lengths /= 2
            print("Reducing step length: {}".format(lengths))
        # If we've obtained an improvement, do pattern moves
        else:
            p, fp = pattern(current_point, new_point, f(new_point), f, lengths)
            while not np.array_equal(p, new_point):
                # new point becomes current, while new point is the output
                # of "pattern" which is the best between pattern, pattern + exp
                # and the old new_point.
                current_point = new_point
                new_point = p
                p, fp = pattern(current_point, new_point,
                                f(new_point), f, lengths)
            current_point = p
        i += 1
    print("<"*40 + " Final Results " + ">" * 40)
    print("Final Point: {}".format(current_point))
    print("Final Value: {}".format(f(current_point)))
    return current_point


if __name__ == "__main__":
    print(hooke_jeeves((0, 0), (1, 1), (0.25, 0.25), func))
