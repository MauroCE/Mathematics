import numpy as np
from collections import OrderedDict
from combinatorics import combinations


class Urn:
    """
    Class implementing an Urn object. This will be used to solve the classic
    Urn problems in probability.
    """
    def __init__(self, **kwargs: dict):
        """
        Constructor of the Urn class. Instantiates the Urn class.

        :param kwargs: Keyword arguments that describe the number of different
                       item types and their number. For instance, if in an urn
                       we have 3 blue balls and 4 red balls, we would call
                       urn = Urn(blue=3, red=4) or something similar.
        :param kwargs: dict
        :return: Nothing to return
        :rtype: None
        """

        # Instantiate other attributes & order kwargs alphabetically
        self.kwargs = OrderedDict(sorted(kwargs.items()))
        self.tot_items = 0
        self.drawn = None
        self.probs_drawing = []
        # Names and quantity
        self.names = sorted(self.kwargs)
        self.qt = [self.kwargs[name] for name in self.names]
        # Set them as attributes and sum the total
        for index, name in enumerate(self.names):
            setattr(self, name, self.qt[index])
            self.tot_items += self.qt[index]
        # Find probabilities to pick any item type
        self.probs_drawing = np.array([qt / self.tot_items
                                       for qt in self.qt])

    def __repr__(self) -> str:
        """
        Overwrite the way it prints out so that it is clearer what object it is
        If we instantiate urn = Urn(blue=4, red=3) then printing urn will get
        Urn: blue=4, red=3.
        :return: String to print on screen.
        :rtype: str
        """
        msg = "Urn: "
        msg += ", ".join("{0} = {1}".format(name, self.qt[index])
                         for index, name in enumerate(self.names))
        return msg

    def random_draw(self, n):
        """
        This method draws n random items from the Urn. Since it is a PAST
        method, we do this before calculating the probability.

        :param n: Number of items to be drawn.
        :type n: int
        :return:
        """
        self.drawn = n

    def prob_picked_are(self, operator="AND", **kwargs):
        """
        This method is used to find the probability that the items drawn
        satisfy certain properties. Examples:

        >>> urn = Urn(blue=4, red=3)
        >>> urn.random_draw(2)

        Up to now we have drawn 2 items randomly. Then, we want to know what is
        the probability that the drawn items are all 2 blue:

        >>> urn.prob_picked_are(blue=2)

        Or we might want to know the probability that one red and one blue

        >>> urn.prob_picked_are(blue=1, red=1)

        We could find the probability that one red or one blue

        >>> urn.prob_picked_are(blue=1, red=1, operator="OR")


        :param operator: Operator used to join the keyword arguments. That is,
                         if we provide "AND" all the conditions will need to be
                         true, if we provide "OR", one condition suffice.
        :type operator: str
        :param kwargs: Keyword arguments describing what we want to find the
                       probability for. The name of the argument should be one
                       of the type names given in __init__, while the value
                       should be the number of items picked that are of that
                       type.
        :type kwargs: dict
        :return:
        """
        # Passed kwargs need to have same item type as initially provided
        if not set(kwargs.keys()).issubset(set(self.kwargs.keys())):
            raise ValueError(
                "Item names allowed are a subset of those given in __init__."
                "\nAllowed: " + ", ".join(self.kwargs.keys()) +
                "\nReceived: " + ", ".join(kwargs.keys())
            )
        # Denominator always the same
        denominator = combinations(self.tot_items, self.drawn)
        # Change numerator based on operator
        if operator.lower() == "and":
            # With an AND operator, we multiply the results
            numerator = 1
            for key in kwargs.keys():
                numerator *= combinations(self.kwargs[key], kwargs[key])
        elif operator.lower() == "or":
            # With an OR operator, we sum the results
            numerator = 0
            for key in kwargs.keys():
                numerator += combinations(self.kwargs[key], kwargs[key])
        else:
            raise ValueError("Operator must be one of 'AND', 'OR'.")
        return numerator / denominator


if __name__ == "__main__":
    # Example of solving a stack overflow problem found here:
    # https://math.stackexchange.com/questions/1705556/urn-problem-find-probability
    urn = Urn(black=996, white=4)
    urn.random_draw(50)
    print(urn.prob_picked_are(black=50))



