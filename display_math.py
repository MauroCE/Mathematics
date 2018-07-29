UNICODE_SUBSCRIPT_BASE = r"\u208"


def unicode_subscript(n):
    """
    Generator for Python Unicode subscripts.
    :param n:
    :return:
    """
    # Store variables for unicode subscript characters 0 to 9
    # They all start like this, look here:
    # https: // www.fileformat.info / info / unicode / char / 2080 / index.htm
    i = 0
    while i <= n-1:
        yield num_to_unicode(i)
        i += 1


def num_to_unicode(n):
    """
    Number to unicode subscript
    :param n:
    :return:
    """
    if n < 10:
        string = bytes(UNICODE_SUBSCRIPT_BASE + str(n),
                       encoding="utf-8").decode('unicode_escape')
    else:
        string = ""
        for digit in str(n):
            string += bytes(UNICODE_SUBSCRIPT_BASE + str(digit),
                            encoding="utf-8").decode('unicode_escape')
    return string