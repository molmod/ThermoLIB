.. _seclab_ug_transformations:

***************
Transformations
***************

Once a free energy profile/surfaces has been constructed in terms of a collective variable CV, we can transform it *a posteriori* to another free energy profiles in terms of another CV. This can be done in two different ways, depending on the nature of the relation between old and new CV:

-   Deterministic transformation from free energy :math:`F_1(CV)` to free energy profile :math:`F_2(Q)` with a deterministic relation described by the mathematical function :math:`Q=f(CV)`
-   Probabilistic transformation from free energy :math:`F_1(CV)` to free energy profile :math:`F_2(Q)` with a probabilistic relation described by the conditional probability :math:`P(Q|CV)`

.. _seclab_ug_transformation_deterministic:

Deterministic transformation
============================

Once a free energy profile has been constructed in terms of a collective variable CV, it can be transformed *a posteriori* to a free energy profile in terms of a new collective variable *Q* that is mathematical function of the original CV, in other words :math:`Q=f(CV)`. This is based on the transformation of variables in probability theory:

.. math::
    
    F_2(Q) &= F_1(f^{-1}(Q)) - k_B T \log\left[\frac{df^{-1}}{dQ}(Q)\right] \\
           &= F_1(f^{-1}(Q)) + k_B T \log\left[\frac{df}{dCV}(f^{-1}(Q))\right]

To do such transformation in ThermoLIB, we need two things: (1) the original free energy profile (FEP) :math:`F(CV)` and (2) a function :math:`Q=f(CV)` and optionally its derivative (if the derivative is not given, it will be estimated through numerical differentiation) to encode the relation between old and new collective variable. The original FEP is defined by an instance of the :py:class:`BaseFreeEnergyProfile <thermolib.thermodynamics.fep.BaseFreeEnergyProfile>` class (or of its child classes). The function :math:`Q=f(CV)` (and its derivative) can be defined as a simple python routine. To illustrate lets consider the function :math:`Q=\frac{1}{0.9}e^{0.9\cdot CV}`, which we can define in python as follows:

.. code-block:: python
    
    def function(cv):
        return np.exp(0.9*cv)/0.9

    def derivative(cv):
        return np.exp(0.9*cv)

Using this, we can perform the transformation from the old FEP ``fep`` to the new FEP ``fep_new`` as follows:

.. code-block:: python

    fep_new = fep.transform_function(function, derivative=derivative)


.. _seclab_ug_transformation_probabilistic:

Probabilistic transformation
============================

Consider the situation of two collective variables CV and Q which are not deterministically related, in other words there is not a unique relation between the two. However, there is a the stochastic correlation between the two, which means we can determine the conditional probability :math:`P(Q|CV)` expressing the probability to find the system in a state with a specific value of Q given that is already is in a state with specific value of CV. This information can be used to transform the free energy profile :math:`F_1(CV)` towards :math:`F_2(Q)` using Bayes theorem from probability theory:

.. math::

    F_2(Q) &= -k_B T \log\left[\int P(Q|CV)\cdot e^{-\beta F_1(CV)} dCV\right]


To do so, we first determine the conditional probability :math:`P(Q|CV)` using the :py:class:`ConditionalProbability1D1D <thermolib.thermodynamics.condprob.ConditionalProbability1D1D>` class as follows:

.. code-block:: python

    #initialize conditional probability
    condprob = ConditionalProbability1D1D()

    #initialize TrajectoryReaders to extract CV and Q values from trajectory files
    cv_reader = ColVarReader([0])
    q_reader  = ColVarReader([1])

    #Read trajectory files and extract samples for conditional probability
    for i in range(ntraj):
        condprob.process_simulation(
            [('COLVAR_%i.dat' %i, q_reader)],
            [('COLVAR_%i.dat' %i, cv_reader)],
        )

    #Compute conditional probability in given bin ranges
    bins_cv = np.arange(start, end, width)
    bins_q = np.arange(start, end, width)
    condprob.finish([bins_q], [bins_cv])

Finally, the original FEP can be tranformed to the new FEP as follows:

.. code-block:: python

    fep_new = condprob.transform(fep)

.. note::

    One can verify that even if there is a deterministic relation :math:`Q=f(CV)`, but we would still do the probabilistic transformation, we get the same result as if we would have done the deterministic transformation. 
    
    Mathematically, this can be proven by realizing that a deterministic relation of the form :math:`Q=f(CV)` can be interpreted as an extreme case of stochastic correlation with conditional probability given by :math:`P(Q|CV)=\delta(Q-f(CV))`, in which :math:`\delta` represents the Dirac delta function/distribution.
    
    Numerically this can be illustrated by using the two approaches given above using the routines implemented in ThermoLIB.
