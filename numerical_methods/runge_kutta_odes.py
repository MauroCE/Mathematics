# External Libraries
import numpy as np
from numpy import polyfit
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import fsolve


# Create two dictionaries containing the gamma, omega and epsilon parameters
# for stiff and non-stiff cases
nonstiff_params = {
    'gamma': -2.0,
    'omega': 5.0,
    'epsilon': 0.05
}
stiff_params = {
    'gamma': -2e5,
    'omega': 20.0,
    'epsilon': 0.5
}

# Values for Explicit problem
exp = {
    'q_start': np.array([[np.sqrt(2.0)],
                         [np.sqrt(3.0)]]),
    't_start': 0.0,
    'dt': 0.05,
    't_end': 1.0
}

# Parameters for plotting
plotting_params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (19, 7),
    'font.family': 'Times New Roman',
    'axes.grid': True,
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}

# Update the plotting parameters.
rcParams.update(plotting_params)


def test_parameters(params):
    """This function tests if the parameters provided in the two dictionaries
    `stiff_params` and `nonstiff_params` are of the correct data type (float).
    It also checks that they are not NaN values, that they're finite and that
    they are real. Basically the checks implemented here are the same as those
    performed to time values in `check_time_inputs()`.
    This function can be useful if you want to type down the parameter values
    quickly so you don't need to write down the `.0` at the end.

    Parameters
    ----------
    params : dict
             Dictionary of parameters that need to be tested.

    Returns
    -------
    None
        Nothing to return. Raises an AssertionError if parameter values are not
        suitable.
    """
    # Store values in a list to make code shorter and checks easier.
    values = list(params.values())
    assert np.all([type(i) == float for i in values]), \
        "Dictionaries containing parameters must have float values for every" \
        "key."
    assert np.all(np.isreal(values)), \
        "Parameter values must be all real."
    assert np.all(np.isfinite(values)), \
        "Parameter values must all be finite."


def check_time_inputs(*args):
    """This function checks that its inputs "args" are sensible time values.
    Time values must be non-negative, finite and must not be NaN values.
    In order to do check the last two conditions this function checks that
    the values are not np.inf or np.nan.
    Notice that the way this function is implemented allows it to be used
    with an iterable input such as a np.array. In other words, this function
    is being overloaded as it usually happens in MATLAB. This choice was taken
    to minimize code duplication.
    This function returns a tuple, except when the tuple contains only one
    element (i.e. when you provide one value as input, whether that is a
    number or an array), in that case it returns the element itself. This
    choice was taken to keep code simpler and not having to deal with
    difficult unpacking: https://stackoverflow.com/a/3721498/6435921 or with
    code duplication.

    Parameters
    ----------
    args : float (or np.array)
           Values that we are expecting to be sensible time values. Although
           floats are preferred, a np.array works as well.

    Returns
    -------
    float
         Args in the same order as provided, after they've been checked. Notice
         that the type will be `tuple` if more than one argument is provided,
         or it can be `np.array` if a np.array input is provided.
    """
    # Check if they are all real, finite, not NaNs and non-negative.
    assert np.all(np.isreal(args)), "Time values must be real."
    assert np.all(np.isfinite(args)), "Time values must be finite."
    assert np.any(~np.isnan(args)), "Time values cannot be NaN."
    assert np.all(np.array(args) >= 0), "Time values must be non-negative."
    # Return a tuple or a single element / np.array
    return args[0] if len(args) == 1 else args


def check_column_vector_inputs(vector, expected_shape=(2, 1)):
    """This function checks whether the input `vector` is an appropriate column
    vector. This function is necessary throughout the whole script because
    the assumption made by both rk3_step and grrk3_step is that the numpy
    arrays are column vectors.
    This function also checks that the vectors are real-valued.
    Important: This function does not throw errors for NaN values or Infinite
    values (np.Nan and np.Inf respectively). This choice was made in order to
    allow the rk3 solution in the stiff case to blow up.
    Notice also that this function does not allow every iterable. For instance
    dict, set and generator data types will not work. This is done to minimize
    code and make everything more readable. It would be an over-kill to check
    for all iterators: https://stackoverflow.com/a/1952481/6435921

    Parameters
    ----------
    vector : np.array
             Object we want to check being a real-valued column vector.
    expected_shape : tuple
                     Shape that we expect the vector to have. In the case of
                     this coursework all vectors should have shape (2, 1).

    Returns
    -------
    np.array
            Correctly-shaped real-valued vector.
    """
    # Vectors must be list, tuple or np.array, no other iterables are allowed
    assert isinstance(vector, (list, tuple, np.ndarray)), \
        "Vectors must be lists, tuples or np.arrays"
    # Vectors must be real-valued
    assert np.all(np.real(vector)), "Arrays must be real-valued."
    # Finally want to work consistently, so transform to column np.array
    vector = np.array(vector).reshape(-1, 1)
    # Check the expected shape
    assert vector.shape == expected_shape, \
        "Column vector shape did not match expected shape. " \
        "Found {} but expected {}".format(vector.shape, expected_shape)
    return vector


def check_inputs(func, t, qn, dt):
    """This function is a wrapper around check_column_vector_inputs and
    check_time_inputs. It is used to check that all the inputs to rk3_step
    and grrk3_step (apart from omega, gamma and epsilon) are of the correct
    type. For a good reference to check when something is callable or a
    function see: https://stackoverflow.com/a/624939/6435921

    Parameters
    ----------
    func : callable
           Function used in rk3_step and grrk3_step. In the coursework it will
           generally be f, defined below.
    t : float
        time used in the algorithm's step. Usually denoted t_n.
    qn : np.array
         Data used by the algorithm, q_n. It will be updated to q_{n+1} by
         either rk3_step or grrk3_step.
    dt : float
         Time delta used in the algorithm.

    Returns
    -------
    tuple
         Arguments func, t, qn, dt in this order, after they've been checked
         and transformed appropriately.
    """
    # Times t, dt need to be non-negative, not NaN, not Inf and real.
    t, dt = check_time_inputs(t, dt)
    # f must be a function, or callable object
    assert callable(func), "Argument f must be a function (or callable)."
    # qn must be a list, a tuple or a np.array, no other iterable is allowed
    qn = check_column_vector_inputs(qn)
    # Return t and dt to be float (in case they were integers)
    t, dt = float(t), float(dt)
    # Return corrected inputs
    return func, t, qn, dt


def f(t, q, **kwargs):
    """
    This function represents the RHS of the ODE that we want to solve. In our
    particular case the function is defined to be as a function of the form:
    a*b - c
    where a is a 2x2 matrix and b, c are 2x1 vectors. They are defined in the
    coursework sheet, equation (2).

    Parameters
    ----------
    t : float
        Time used to evaluate the function. It will be passed by `rk3_step()`
        or by `grrk3_step()` function. In the coursework sheet this is denoted
        t_n.
    q : np.array
        Current approximate solution q_n or the initial data q_0. Notice it
        will contain float values.
    kwargs : dict
             Keyword arguments for `f`. In our example they will be gamma,
             omega and epsilon, which are provided in `stiff_params` and
             `nonstiff_params` dictionaries defined at the top of the file.
             The type of the dictionaries will be `Dict[str, float]`.

    Returns
    -------
    np.array
            Result of applying RHS of the ODE to the input arguments provided.
    """
    # Check t is a correct time value, and check q is a suitable column vector
    t, q = check_time_inputs(t), check_column_vector_inputs(q)
    # Check that all other parameters t, gamma, omega and epsilon are scalars
    try:
        # allow mis-spelling by using capital letters.
        kwargs = {key.lower(): float(kwargs[key]) for key in kwargs.keys()}
    except TypeError:
        raise TypeError(
            "t, gamma, omega and epsilon should be of type float. Found {} "
            "respectively".format([type(p).__name__ for p in kwargs.values()])
        ) from None  # Providing a more helpful message by overwriting
    # Check whether kwargs have correct naming
    assert set(kwargs.keys()) == {'gamma', 'omega', 'epsilon'}, \
        "Incorrect naming of parameters. Expecting 'gamma', 'omega' and " \
        "'epsilon'. Received {}".format([", ".join(k) for k in kwargs.keys()])
    # Remember q(t) = (x(t), y(t))^T . Get x and y to simplify calculations
    x, y = q[0, 0], q[1, 0]
    # Define the 2x2 matrix containing gamma and epsilon. Lowercase for PEP8
    gamma, omega, epsilon = kwargs['gamma'], kwargs['omega'], kwargs['epsilon']
    a = np.array([[gamma, epsilon],
                  [epsilon, -1]])
    # Define 2x1 vector b multiplying on the right matrix a
    b = np.array([[(-1.0 + x**2.0 - np.cos(t)) / (2.0*x)],
                  [(-2.0 + y**2.0 - np.cos(omega*t)) / (2.0*y)]])
    # Define 2x1 vector c subtracted to a*b
    c = np.array([[np.sin(t) / (2.0*x)],
                  [omega*np.sin(omega*t) / (2.0*y)]])
    return np.dot(a, b) - c


def exact_solution(t, omega):
    """
    Exact analytical solution for the ODE. Ideally we want to compare this
    against rk3 and grrk3 algorithms. Notice that since we are only using this
    function for plotting purposes, there is no need in outputting a 2x1 column
    vector. Rather, we can just return a tuple where the first element is the
    first coordinate and the second element is the second coordinate. This way,
    it is easier to unpack them and plot them.
    Notice that this function is given by the equation (3) in the coursework
    sheet.

    Parameters
    ----------
    t : np.array
        Values of t used to evaluate the exact solution. Ideally, they should
        come from a call to `np.linspace()`. Elements of this array are of
        float data type.
    omega : float
            Omega parameter contained in `f` function. This is the same
            parameter found in `stiff_params` and `nonstiff_params`
            dictionaries.

    Returns
    -------
    tuple
         Solution evaluated at `t` and `omega` provided. The first element of
         the tuple will be the result for the first coordinate, whereas the
         second (and last) element of the tuple will be the result for the
         second coordinate.
    """
    # Check omega input
    assert np.all(np.isreal(omega)) and np.isfinite(omega), \
        "Omega must be real and finite."
    # Check time input
    t = check_time_inputs(t)
    # "X" element, in 1x1 position
    first_coordinate = np.sqrt(1 + np.cos(t))
    # "Y" element, in 2x1 position
    second_coordinate = np.sqrt(2 + np.cos(omega*t))
    return first_coordinate, second_coordinate


def rk3_step(func, t, qn, dt, **kwargs):
    """
    Single step of the explicit rk3 algorithm. It performs one step in order to
    obtain the (n+1)th approximate solution q_{n+1} from the nth approximate
    solution q_{n}. Notice this does not perform the whole algorithm. In order
    to perform the whole algorithm  one needs to use the `repeat()` function
    with parameter `rk` set to `True`.
    Notice this function checks that the provided inputs are correct before
    using them and feeding them to `func`.
    Notice that this function implements equations (4a)-(4d) in the coursework
    sheet provided.

    Parameters
    ----------
    func : callable
           Function representing the RHS of the ODE that we want to solve.
    t : float
        Time step t_n used for this step of the algorithm.
    qn : np.array
         Data of the previous iteration (or initial data in the first
         iteration). Denoted q_n. Elements of this array will have float
         data type
    dt : float
         Time delta used to go from t_n to t_{n+1} in the `repeat()` function.
    kwargs : dict
             keyword arguments for func. In our example they will be gamma,
             omega and epsilon.

    Returns
    -------
    np.array
            Approximate solution q_{n+1}. This is again a 2x1 column vector in
            our specific case. In general, it will have the same shape of the
            input argument `qn`. Elements of this array will have float data
            type.
    """
    # Check inputs have correct types and dimensions
    func, t, qn, dt = check_inputs(func, t, qn, dt)
    # Find k1, k2, k3 and k4 following the formulas on the sheet
    k1 = func(t, qn, **kwargs)
    # Stop execution if function `func` does not return arrays with the same
    # shape as qn. This is done in case we want to use a different `f`, which
    # does not return output arrays of the same shape as its input arrays
    assert k1.shape == qn.shape, \
        "Callable argument `func` must output arrays of the same shape " \
        "as `qn`."
    k2 = func(t + 0.5 * dt, qn + dt * 0.5 * k1, **kwargs)
    k3 = func(t + dt, qn + dt * (-k1 + 2.0 * k2), **kwargs)
    # Finally return the new approximate solution
    return qn + dt*(k1 + 4.0*k2 + k3) / 6.0


def grrk3_step(func, t, qn, dt, **kwargs):
    """
    Single step of the implicit grrk3 algorithm. It performs one step in order
    to obtain the (n+1)th approximate solution q_{n+1} from the nth
    approximate solution q_{n}. Notice this does not perform the whole
    algorithm. In order to perform the whole algorithm  one needs to use the
    `repeat()` function with parameter `rk` set to `True`.
    Notice this function checks that the provided inputs are correct before
    using them and feeding them to `func`.
    This implements the algorithm described in equations (5a)-(5c) using the
    method in equations (6) and (7) of the coursework worksheet.

    Parameters
    ----------
    func : callable
           Function representing the RHS of the ODE that we want to solve.
    t : float
        Time step t_n used for this step of the algorithm.
    qn : np.array
         Data of the previous iteration (or initial data in the first
         iteration). Denoted q_n. Elements of this array will have float
         data type.
    dt : float
         Time delta used to go from t_n to t_{n+1} in the `repeat()` function.
    kwargs : dict
             keyword arguments for func. In our example they will be gamma,
             omega and epsilon.

    Returns
    -------
    np.array
            Approximate solution q_{n+1}. This is again a 2x1 column vector in
            our specific case. In general, it will have the same shape of the
            input argument `qn`. Elements of this array will have float
            data type.
    """
    # Check inputs have correct types and dimensions
    func, t, qn, dt = check_inputs(func, t, qn, dt)
    # Find initial guess for K = (k1, k2)^T
    k1 = func(t + dt / 3.0, qn, **kwargs)
    # Stop execution if function `func` does not return arrays with the same
    # shape as qn. This is done in case we want to use a different `f`, which
    # does not return output arrays of the same shape as its input arrays
    assert k1.shape == qn.shape, \
        "Callable argument `func` must output arrays of the same shape " \
        "as `qn`."
    k2 = func(t + dt, qn, **kwargs)
    k = np.vstack((k1, k2))

    def f_capital(k_new):
        """
        Function F(K) for which we want to find a root F(K)=0. Notice that
        input argument has "_new" subscript to avoid PEP8 errors due to
        variable name already assigned in the outer scope. (i.e. the scope
        of `grrk3_step()`). Notice that one of the reasons for this function to
        be defined inside of `grrk3_step()` is that in this way it can use
        **kwargs. Indeed, we need to pass both stiff and non-stiff parameters
        to F because `f` is embedded in it. However, using `fsolve` we cannot
        pass keyword arguments.

        Parameters
        ----------
        k_new : np.array
                Vector containing `k1` and `k2`, i.e. K = (k1, k2)^T.

        Returns
        -------
        np.array
                Returns F(K) as a flattened array. This is done because SciPy
                function `fsolve` cannot work with multi-dimensional arrays.
        """
        # Check that qn is a correct column vector. This is check should never
        # fail because q_n was checked at the top level `grrk3_step()`.
        qn_new = check_column_vector_inputs(qn)
        # Notice that k_new will have to have shape (2*len(qn), ) no matter
        # the output shape of f, because we are stacking them vertically and
        # then feeding them as k.ravel()
        assert k_new.shape == (2*len(qn), ), \
            "k_new must be a column vector with 2 * n number of rows, where" \
            " n is the number of rows of qn column vector."
        # Unpack for simplicity. k1_new and k2_new are now column vectors.
        # This is done so that we can feed them into f:R^2->R^2
        k1_new = k_new[:len(qn_new)].reshape(qn_new.shape)
        k2_new = k_new[len(qn_new):].reshape(qn_new.shape)
        # Find function output to be subtracted from k1 ravelled
        fx_t = t + dt / 3.0  # t input for f
        fx_q = qn_new + (dt / 12.0) * (5.0 * k1_new - k2_new)  # q input
        fx = func(fx_t, fx_q, **kwargs)
        # Find function output to be subtracted from k2 ravelled
        fy_t = t + dt
        fy_q = qn_new + 0.25 * dt * (3.0 * k1_new + k2_new)
        fy = func(fy_t, fy_q, **kwargs)
        # Subtract fx from k1 and fy from k2. Put them in a ravelled array
        return np.vstack((k1_new - fx, k2_new - fy)).ravel()

    # Feed it into fsolve to find new solution. Using np.array() in order to
    # suppress PEP8 inspection errors due to duck typing or dynamic dispatch
    sol = np.array(
        fsolve(f_capital, k.ravel())
    ).reshape(k.shape)
    # Now find q_{n+1}
    k1 = sol[:len(qn)]
    k2 = sol[len(qn):]
    return qn + 0.25*dt*(3.0*k1 + k2)


def repeat(func, q_start, t_start, dt, t_end, rk=True, **kwargs):
    """
    This function repeats either `rk3_step()` or `grrk3_step()` functions.
    Which algorithm is used is chosen via `rk` keyword argument. If `True`,
    then rk3 algorithm will be performed, otherwise the grrk3_algorithm will be
    performed. This function does not check whether the provided inputs are
    correct or not because it uses `rk3_step()` and `grrk3_step()` which
    already have error checking procedures, so this is done in order to not
    duplicate code and to make the algorithm more efficient. Notice that this
    does not affect the robustness of the algorithm because if there is an
    issue with the inputs, this will be found in lower-level functions.

    Parameters
    ----------
    func : callable
           Function representing the RHS of the ODE to be solved.
    q_start : np.array
              Initial data used in the first iteration to get q_1.
    t_start : float
              Initial time used to run algorithm at the first iteration.
    dt : float
         Initial time delta used to run algorithm at the first iteration.
    t_end : float
            Final time used as stopping criteria. The `repeat()` function will
            stop when the current time step + `dt` will exceed `t_end`.
    rk : bool
         If `True`, `rk3_step()` will be used as algorithm step, otherwise
         `grrk3_step` will be used.
    kwargs : dict
             Extra parameters for f. In this case gamma, omega, epsilon. In our
             case it will be of type Dict[str, float].

    Returns
    -------
    np.array
            Results from running the algorithm until the stopping criteria.
            Each result will be stored as a column vector in a 2D array of
            shape (2, n) where n is the number of iterations required by the
            algorithm. Therefore the solution for step i will be result[:, i].
    """
    # Decide whether to use rk3_step or grrk3_step
    algorithm = rk3_step if rk else grrk3_step
    t, q = t_start, q_start
    # Check that t_end is time input
    t_end = check_time_inputs(t_end)
    n_iter = int((t_end - t_start) / dt) + 1
    # Store all solutions found
    results = np.zeros((2, n_iter))
    # First solution is the initial data
    results[:, 0] = q_start.flatten()
    # Algorithm stops at the end of the time interval
    for i in range(1, n_iter):
        q = algorithm(func, t, q, dt, **kwargs)
        results[:, i] = q.flatten()
        t += dt
    return results


def convergence(func, n_dt, num=0.1, rk=True, **kwargs):
    """This function answers question 4. It takes a function `func`
    representing RHS of ODE to solve, an algorithm choice (via `rk=True`) and
    repeats the algorithm chosen for `func` using `dt` as a time delta ranging
    from `t_start` up to `t_end`. At each step, calculates the error with
    respect to the exact solution in the y coordinate and appends it to an
    empty array. At the end, it returns the errors together with a numpy array
    containing all the dt values used. These dt values will then be used for
    plotting purposes.
    If you want to plot errors against dt, you only need to plot the first
    output of this function as x-values and the second output as y-values.

    Parameters
    ----------
    func : callable
           Function implementing the RHS of the ODE that we want to solve.
           In our case this will always be defined by the function `f`.
    n_dt : int
           Number of dt to use as 0.1/2**j (so j is in (0, n-1)).
    num : float
          Numerator used in finding the dt values. In the non-stiff case we
          used 0.1 whereas for the stiff case we used 0.05.
    rk : bool
         If True, we use rk3, otherwise grrk3. This parameter is passed to the
         repeat function.
    kwargs : dict
             Extra parameters for f. In this case gamma, omega, epsilon. In our
             case it will be of type Dict[str, float].

    Returns
    -------
    tuple
         dt_values, errors. The first output `dt_values` is an array containing
         the values of `dt` used to evaluate the algorithm. The second output
         is an array containing the errors in the `y` variable of the algorithm
         of choice against the exact solution.
    """
    dt_values = np.array([num / 2.0**j for j in range(n_dt)])
    # create a vector to store errors
    errors = np.zeros(n_dt)
    for i, dt in enumerate(dt_values):
        # Get the full results (x(t), y(t))^T using alg
        alg_sol = repeat(func, exp['q_start'], exp['t_start'],
                         dt, exp['t_end'], rk=rk, **kwargs)
        # Obtain the same, but using the exact solution
        n_t = int((exp['t_end'] - exp['t_start']) / dt) + 1
        t_exact = np.linspace(exp['t_start'], exp['t_end'], n_t)
        exact_sol = exact_solution(t_exact, kwargs['omega'])
        # Append the error to the array
        errors[i] = dt * np.abs(alg_sol[1, :].flatten() - exact_sol[1]).sum()
    return dt_values, errors


def plot_nonstiff():
    """This function plots rk3 and grrk3's x and y components against a
    vector of time steps, for the non-stiff case. These solutions are plotted
    together with the exact solution for a comparison.This is the first plot
    requested in the coursework.

    Returns
    -------
    None
        Nothing to return. A plot is displayed to screen.
    """
    # Find Exact Solution
    exact_timesteps = np.linspace(0, 1, 1000)
    x_exact, y_exact = exact_solution(exact_timesteps,
                                      omega=nonstiff_params['omega'])
    # Find RK3 solutions
    rk_timesteps = np.linspace(0, 1, 21)
    rk3_solutions = repeat(f, **exp, rk=True, **nonstiff_params)
    x_rk3 = rk3_solutions[0, :]
    y_rk3 = rk3_solutions[1, :]
    # Find GRRK3 solutions
    grrk3_solutions = repeat(f, **exp, rk=False, **nonstiff_params)
    x_grrk3 = grrk3_solutions[0, :]
    y_grrk3 = grrk3_solutions[1, :]
    # Construct a canvas figure and a 1x2 axes array
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Exact and Numerical solutions for Non-stiff System",
                 fontsize=20)
    # First subplot shows the first coordinate ("x", to say)
    axes[0].plot(exact_timesteps, x_exact, label="Exact Solution")
    # axes[0].plot(rk_timesteps, x_rk3, "g-", marker="x", label="RK3")
    axes[0].plot(rk_timesteps, x_rk3, "g x", ms=13, label="RK3")
    # axes[0].plot(rk_timesteps, x_grrk3, "r-",  marker="+", label="GRRK3")
    axes[0].plot(rk_timesteps, x_grrk3, "r +", ms=13, label="GRRK3")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$x$")
    axes[0].set_title("X-coordinate Plot")

    # Second subplot shows the second coordinate ("y", to say)
    axes[1].plot(exact_timesteps, y_exact, label="Exact Solution")
    # axes[1].plot(rk_timesteps, y_rk3, "g-", marker="x", label="RK3")
    axes[1].plot(rk_timesteps, y_rk3, "g x", ms=13, label="RK3")
    # axes[1].plot(rk_timesteps, y_grrk3, "r-", marker="+", label="GRRK3")
    axes[1].plot(rk_timesteps, y_grrk3, "r +", ms=13, label="GRRK3")
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_title("Y-coordinate Plot")
    # Add a single legend
    plt.legend()  # Or fig.legend()
    # Finally, make sure everything is readable by using tight layout
    plt.tight_layout()
    plt.show()


def plot_stiff_rk3():
    """This function plots rk3 algorithm's results together with the exact
    solution. This plot shows how rk3 algorithm blows up in the stiff case.
    Since the coursework description wasn't very clear I've chosen the axes
    limits to restrict to the scale of the exact solution. However, it is
    possible to restrict these even more, to appreciate better when the rk3
    solution blows up.

    Returns
    -------
    None
        Nothing to return. A plot is displayed to screen.
    """
    # Find Exact Solution
    exact_timesteps = np.linspace(0, 1, 1001)
    x_exact, y_exact = exact_solution(exact_timesteps,
                                      omega=stiff_params['omega'])
    # Find RK3 solutions
    rk_timesteps = np.linspace(0, 1, 1001)
    rk3_solutions = repeat(f,
                           q_start=exp['q_start'],
                           t_start=exp['t_start'],
                           dt=0.001,
                           t_end=exp['t_end'],
                           rk=True,
                           **stiff_params)
    x_rk3 = rk3_solutions[0, :]
    y_rk3 = rk3_solutions[1, :]
    # Figure
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Exact and RK3 Numerical solutions for Stiff System",
                 fontsize=20)
    # First subplot shows the first coordinate ("x", to say)
    axes[0].plot(exact_timesteps, x_exact, label="Exact Solution")
    axes[0].plot(rk_timesteps, x_rk3, "g-", ms=13, label="RK3")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$x$")
    axes[0].set_title("X-coordinate Plot")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(1.2, 1.5)  # 0.035
    # Second subplot shows the second coordinate ("y", to say)
    axes[1].plot(exact_timesteps, y_exact, label="Exact Solution")
    axes[1].plot(rk_timesteps, y_rk3, "g-", ms=13, label="RK3")
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_title("Y-coordinate Plot")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(1.0, 1.8)
    # Add a single legend, make sure everything is readable and show plot
    plt.legend()  # Or fig.legend()
    plt.tight_layout()
    plt.show()


def plot_stiff_grrk3():
    """This function plots grrk3 solution together with the exact solution
    for the stiff case, showing that grrk3 is stable.

    Returns
    -------
    None
        Nothing to return. A plot is displayed to screen.
    """
    # Find Exact Solution
    exact_timesteps = np.linspace(0, 1, 1001)
    x_exact, y_exact = exact_solution(exact_timesteps,
                                      omega=stiff_params['omega'])
    # Find RK3 solutions
    rk_timesteps = np.linspace(0, 1, 201)
    grrk3_solutions = repeat(f,
                             q_start=exp['q_start'],
                             t_start=exp['t_start'],
                             dt=0.005,
                             t_end=exp['t_end'],
                             rk=False,
                             **stiff_params)
    x_grrk3 = grrk3_solutions[0, :]
    y_grrk3 = grrk3_solutions[1, :]
    # Figure
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Exact and GRRK3 Numerical solutions for Stiff System",
                 fontsize=20)
    # First subplot shows the first coordinate ("x", to say)
    axes[0].plot(exact_timesteps, x_exact, label="Exact Solution")
    axes[0].plot(rk_timesteps, x_grrk3, "g x", ms=13, label="GRRK3")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$x$")
    axes[0].set_title("X-coordinate Plot")
    # Second subplot shows the second coordinate ("y", to say)
    axes[1].plot(exact_timesteps, y_exact, label="Exact Solution")
    axes[1].plot(rk_timesteps, y_grrk3, "g x", ms=13, label="GRRK3")
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_title("Y-coordinate Plot")
    # Add a single legend, make sure everything is readable and show plot
    plt.legend()  # Or fig.legend()
    plt.tight_layout()
    plt.show()


def plot_convergence_nonstiff(n_dt, **kwargs):
    """Plots convergence of rk3 and grrk3 in the non-stiff case, showing that
    both algorithms converge. In order to obtain a sensible plot, a log-log
    scale is used here. This function therefore plots the log of the errors
    against the log of dt.
    A very delicate step is that of the best fit line. For both rk3_step and
    grrk3_step we find the best-fit coefficients that fit the data given by
    (log(dt), log(error))_i. The curve that has been fit is a straight line
    because by looking at the log-log plot the relationship looked linear.
    Notice that plotting the best fit line is a delicate step because, having
    found slope and intercept, we need to plot the following:
    y = (dt**slope) * exp(intercept)
    In this way when we plot a log log plot we have:
    log(y) = slope * log(dt) + intercept
    Which is the correct way of obtaining the line, as slope and intercept were
    found by fitting log(dt) and log(errors), not dt and errors.

    Parameters
    ----------
    n_dt : int
           Number of time deltas to use as 0.1/2**j (so j is in (0, n-1))
    kwargs : dict
             Non-stiff parameters, in this coursework they will be
             nonstiff_params.

    Returns
    -------
    None
        Nothing to return. A plot is displayed to screen.
    """
    # Get errors both for rk3 and grrk3
    dt, rk3_errors = convergence(f, n_dt, num=0.1, rk=True, **kwargs)
    _, grrk3_errors = convergence(f, n_dt, num=0.1, rk=False, **kwargs)
    # Find best-fit coefficients for both. Fit a line as pattern seems linear.
    # Since we want to use a log-log plot, we fit the coefficients to that.
    # Otherwise, we would end up with coefficients fitting dt and rk3_errors
    # for instance.
    rk3_fitted = polyfit(np.log(dt), np.log(rk3_errors), deg=1)
    rk3_slope, rk3_int = rk3_fitted[0], rk3_fitted[1]
    grrk3_fitted = polyfit(np.log(dt), np.log(grrk3_errors), deg=1)
    grrk3_slope, grrk3_int = grrk3_fitted[0], grrk3_fitted[1]
    # Because we want to do a Log-Log plot (due to the very definition of the
    # dts) we need to keep the coefficients the same. To do this, we take the
    # exp() of all the slopes. Because we want to have:
    # log(exp(a)* t**b) = log(exp(a)) + log(t**b) = a + b log(t)
    rk3_int, grrk3_int = np.exp(rk3_int), np.exp(grrk3_int)
    # Plot them in the same figure
    fig, ax = plt.subplots()
    ax.loglog(dt, rk3_errors, "r +", ms=13, label="RK3")
    ax.loglog(dt, grrk3_errors, "b x", ms=13, label="GRRK3")
    # Recall that log(t^b)=b log(t) and log(ab) = log(a) + log(b)
    ax.loglog(dt, (dt**rk3_slope)*rk3_int, "k-", label="RK3 fitted-line")
    ax.loglog(dt, (dt**grrk3_slope)*grrk3_int, "g-", label="GRRK3 fitted-line")
    ax.set_ylabel(r"$\log(E_j)$")
    ax.set_xlabel(r"$\log(\Delta t)$")
    ax.set_title("Log-Log Plot of Convergence of RK3 and "
                 "GRRK3 in non-stiff case")
    # Annotate the slopes to show algorithm is converging at third order.
    # One critique that could be raised would be that the position of these
    # annotations is very specific and would not work for other plots. However,
    # notice that this is true also for the axes limits, which are required
    # by the coursework instructions. Therefore if `f` or some other element
    # were to change, one could simple comment these annotations out or adjust
    # their `xy` and `xytext` positions. Alternatively, one could add the slope
    # information into the legend.
    ax.annotate(
        s="Slope {:.3}".format(rk3_slope),
        xy=(0.00882807, 4.49471e-07),
        xytext=(0.00397601, 1.70438e-05),
        arrowprops=dict(arrowstyle='->', color="m")
    )
    ax.annotate(
        s="Slope {:.3}".format(grrk3_slope),
        xy=(0.0106476, 3.0743e-07),
        xytext=(0.0129662, 1.31199e-10),
        arrowprops=dict(arrowstyle='->', color="m")
    )
    # Set some axis limit to visualize the plot better
    ax.set_xlim((1/2)*1e-3, 1.0/7.5)
    ax.set_ylim(1e-11, 1e-2)
    plt.legend()
    plt.show()


def plot_convergence_stiff(n_dt, **kwargs):
    """
    Plot showing convergence of grrk3 algorithm in the stiff case.

    Parameters
    ----------
    n_dt : int
           Number of time deltas to use as 0.1/2**j (so j is in (0, n-1))
    kwargs : dict
             Non-stiff parameters, in this coursework they will be
             nonstiff_params.

    Returns
    -------
    None
        Nothing to return. A plot is displayed to screen.
    """
    # Get errors both for grrk3
    dt, grrk3_errors = convergence(f, n_dt, num=0.05, rk=False, **kwargs)
    # Find best-fit coefficients for both. Fit a line as pattern seems linear.
    # Since we want to use a log-log plot, we fit the coefficients to that.
    # Otherwise, we would end up with coefficients fitting dt and rk3_errors
    # for instance.
    grrk3_fitted = polyfit(np.log(dt), np.log(grrk3_errors), deg=1)
    grrk3_slope, grrk3_int = grrk3_fitted[0], grrk3_fitted[1]
    # Because we want to do a Log-Log plot (due to the very definition of the
    # dts) we need to keep the coefficients the same. To do this, we take the
    # exp() of all the slopes. Because we want to have:
    # log(exp(a)* t**b) = log(exp(a)) + log(t**b) = a + b log(t)
    grrk3_int = np.exp(grrk3_int)
    # Plot them in the same figure
    fig, ax = plt.subplots()
    ax.loglog(dt, grrk3_errors, "b x", ms=13, label="GRRK3")
    # Recall that log(t^b)=b log(t) and log(ab) = log(a) + log(b)
    ax.loglog(dt, (dt**grrk3_slope)*grrk3_int, "g-", label="GRRK3 fitted-line")
    ax.set_ylabel(r"$\log(E_j)$")
    ax.set_xlabel(r"$\log(\Delta t)$")
    ax.set_title("Log-Log Plot of Convergence of GRRK3 in Stiff Case")
    # Annotate the slopes to show algorithm is converging at third order
    # Similar remarks hold here as for `plot_convergence_nonstiff()`.
    ax.annotate(
        s="Slope {:.3}".format(grrk3_slope),
        xy=(0.00454807, 4.02735e-06),
        xytext=(0.00675, 3.69966e-08),
        arrowprops=dict(arrowstyle='->', color="m")
    )
    plt.legend()
    plt.show()


def test_f():
    """This function tests the execution of the function f provided in the
    coursework. We can use `np.array_equal()` because the expected solution is
    very simple.
    If we use t=0, qn = [1, 1] and omega=gamma=epsilon=1 our
    matrix-vector calculation should be:
    (1  1)  *  (-1/2)  -  (0)  =  (-3/2)
    (1 -1)     (-1)       (0)     (1/2)
    therefore our test function should return np.array([[-1.5], [0.5]]).
    Furthermore, it also checks that you can pass tuples and lists as qn.

    Returns
    -------
    None
        Nothing to return. It will raise an AssertionError if the test fails.
    """
    simple_params = {
        'gamma': 1,
        'epsilon': 1,
        'omega': 1
    }
    # Test if it gives the correct calculation
    result = f(0, np.array([[1], [1]]), **simple_params)
    assert np.array_equal(result, np.array([[-1.5], [0.5]])), \
        "Calculations test for f failed."
    # Test if it works with lists
    result_list = f(0, [1, 1], **simple_params)
    assert np.array_equal(result, result_list), "List-input test for f failed."
    # Test if it works with tuples
    result_tup = f(0, (1, 1), **simple_params)
    assert np.array_equal(result, result_tup), "Tuple-input test for f failed."


def test_exact_solution():
    """This function tests whether exact_solution is implemented correctly.
    There is no need for using `np.isclose()` because the exact solution
    is very simple, so we can use `np.array_equal()`.

    Returns
    -------
    None
        Nothing to return. It will raise an AssertionError if the test fails.
    """
    # Test basic calculations for t=0 and omega=1
    exact = np.array(exact_solution(t=0, omega=1))
    expected = np.array([np.sqrt(2.0), np.sqrt(3.0)])
    assert np.array_equal(exact, expected), "Exact Solution test1 failed."
    # Test basic calculations for t=np.pi and omega=1
    exact = np.array(exact_solution(np.pi, 1))
    expected = np.array([0, 1])
    assert np.array_equal(exact, expected), "Exact Solution test2 failed."


def test_rk3_step():
    """
    This function tests that rk3_step performs simple operations correctly.
    The exact result has been found by applying each operation separately
    on the Python console. Notice that we allow some discrepancy between
    the grrk3_step result and the exact result because of round-off errors
    or truncation errors.
    If no discrepancy is allowed, then `np.array_equal()` can be used instead.

    Returns
    -------
    None
        Nothing to return. It will raise an AssertionError if the test fails.
    """
    # Get the result and compare it to what it should be
    r = rk3_step(f, 0.0, [np.sqrt(2), np.sqrt(3)], 0.05, **nonstiff_params)
    exact = np.array([[1.4137717014140196], [1.7230469255590304]])
    assert np.all(np.isclose(r, exact, rtol=1e-10, atol=1e-10)), \
        "Test for rk3_step failed."


def test_grrk3_step():
    """This function tests that the grrk3_step function performs correct
    simple calculations. The values
    `np.array([[1.4137716368736974], [1.7230560501188303]])`
    have been found by performing each operation separately using the Python
    console. Notice that we allow some discrepancy between the grrk3_step
    result and the exact result because of round-off errors or truncation
    errors.
    If no discrepancy is allowed, then `np.array_equal()` can be used instead.

    Returns
    -------
    None
        Nothing to return. It will raise an AssertionError if the test fails.
    """
    # Get the result and compare it to what it should be
    r = grrk3_step(f, 0.0, [np.sqrt(2), np.sqrt(3)], 0.05, **nonstiff_params)
    exact = np.array([[1.4137716368736974], [1.7230560501188303]])
    assert np.all(np.isclose(r, exact, rtol=1e-10, atol=1e-10)), \
        "Test for rk3_step failed."


if __name__ == "__main__":
    # Run test for parameters provided
    test_parameters(stiff_params)
    test_parameters(nonstiff_params)
    print("Tests for stiff and non-stiff parameters were successful.")
    # Run tests for exact solution, f, rk3_step and grrk3_step
    test_exact_solution()
    test_f()
    test_rk3_step()
    test_grrk3_step()
    print("Tests for `exact_solution()`, `f`, `rk3_step()` and `grrk3_step()`"
          " were successful.")
    # Run Non-Stiff Plots
    plot_nonstiff()
    plot_convergence_nonstiff(n_dt=8, **nonstiff_params)
    # Run Stiff Plots
    plot_stiff_rk3()
    plot_stiff_grrk3()
    plot_convergence_stiff(n_dt=8, **stiff_params)
