.. _seclab_ug_projection:

******************************
(De)projections of the FEP/FES
******************************

Projecting 2D FES to 1D FEP
===========================

A projection of a 2D Free energy surface towards a 1D free energy profile can be one of the following cases:

- Projecting out one of the collective variables, e.g. going from :math:`F_{12}(CV_1,CV_2)` towards :math:`F_1(CV_1)`
- Projecting to a function of the original two collective variables, i.e. going from :math:`F_2(CV_1,CV_2)` towards :math:`F_1(Q)` with :math:`Q=f(CV_1,CV_2)`

The first case is as easy as applying the :py:meth:`project_cv1 <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_cv1>` or :py:meth:`project_cv2 <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_cv2>` routine:

.. code-block:: python

    fep1 = fes.project_cv1()
    #or
    fep2 = fes.project_cv2()

The second case can be done by application of the :meth:`project_function <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_function>` routine which requires the definition of the function :math:`Q=f(CV_1,CV_2)` in a similar fashion as was the case in transformations:

.. code-block:: python

    #define function
    def function(cv1, cv2):
        return 0.5*(cv2-cv1)**2

    #define range of bins of new collective variable
    qs = np.arange(..., ..., ...)

    #apply projection
    fep = fes.project_function(function, qs)

In case the function is simply the average :math:`Q=0.5\cdot(CV_1+CV_2)` or the difference :math:`Q=CV_2-CV_1`, one can use the already implemented specific routines :py:meth:`project_average <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_average>` and :py:meth:`project_difference <thermolib.thermodynamics.fep.FreeEnergySurface2D.project_difference>`:

.. code-block:: python

    fep_avg = fes.project_average()
    #or
    fep_diff = fes.project_difference()



Deprojecting 1D FEP to 2D FES
=============================

To do a deprojection of a 1D FEP :math:`F_1(CV)` towards a 2D free energy surface :math:`F_{12}(Q_1,Q_2)`, knowledge of only :math:`F_1(CV)` is insufficient and additional information from the original simulation is required. Depending on the situation, multiple approaches to do this are possible. The first, most general, approach is based on Bayes rule

.. math::

    F_{12}(Q_1,Q_2) &= -k_BT\log\left[\int P(Q_1,Q_2|CV)\cdot e^{-\beta F_1(CV)} dCV\right]

and, hence, involves construction of the conditional probability :math:`P(Q_1,Q_2|CV)`. This property represents the probability that the system is in a state with a specific value of Q1 and Q2, given that the system was in a state with specific value of *CV* and can be extracted from the original simulation data using the :py:class:`ConditionalProbability1D2D <thermolib.thermodynamics.condprob.ConditionalProbability1D2D>` class:

.. code-block:: python

    #initialize conditional probability
    condprob = ConditionalProbability1D2D()

    #initialize TrajectoryReaders to extract CV and Q1, Q2 values from trajectory files
    cv_reader = ColVarReader([0])
    q1_reader = ColVarReader([1])
    q2_reader = ColVarReader([2])

    #Read trajectory files and extract samples for conditional probability
    for i in range(ntraj):
        condprob.process_simulation(
            [('COLVAR_%i.dat' %i, q1_reader), ('COLVAR_%i.dat' %i, q2_reader)], 
            [('COLVAR_%i.dat' %i, cv_reader)], 
        )

    #Compute conditional probability in given bin ranges
    bins_cv = np.arange(start, end, width)
    bins_q1 = np.arange(start, end, width)
    bins_q1 = np.arange(start, end, width)
    condprob.finish([bins_q1,bins_q2], [bins_cv])

This conditional probability can be plotted using its :py:meth:`plot <thermolib.thermodynamics.condprob.ConditionalProbability1D2D.plot>` routine. Finally, the 2D free energy surface :math:`F_{12}(Q_1,Q_2)` can now be constructed from the 1D FEP and the conditional probability as follows:

.. code-block:: python

    fes = condprob.deproject(fep)

.. note::

    In case we would like to deproject towards new CVs in which Q1 is just the original CV, i.e. :math:`Q_1=CV`, we can make some simplifications to the Bayes theory resulting in
    
    .. math::

        F(CV,Q) &= F(CV) - k_BT\log\left[P(Q|CV)\right]
        
    
    and use the :py:class:`ConditionalProbability1D1D <thermolib.thermodynamics.condprob.ConditionalProbability1D1D>` class to implement this

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

        #construct deprojected 2D FES
        fes = condprob.deproject(fep)
    
    Finally, as a second alternative method to get the 2D FES in this specific situation, one could also have done a 2D WHAM using the original simulation data biased along one of these CVs as was described in :ref:`SCENARIO 2 of constructing a 2D histogram from WHAM <seclab_ug_2Dhistogram_wham_scenario2>`.

