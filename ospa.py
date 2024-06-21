"""
    Computes the Optimal SubPattern Assignment (OSPA) distance [1] for two sets
    of 'Track' objects. The OSPA distance is measured between two
    point patterns. Main skeleton is taken from 'StoneSoup' documentation [2].

    The OSPA metric is calculated at each time step in which a 'Track'
    object is present

    Reference:
        [1] A Consistent Metric for Performance Evaluation of Multi-Object
        Filters, D. Schuhmacher, B. Vo and B. Vo, IEEE Trans. Signal Processing
        2008
        [2] StoneSoup Documentation: https://github.com/dstl/Stone-Soup/blob/main/stonesoup/metricgenerator
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

class OSPAMetric:

    c: float = 5 #'Maximum distance for possible association' (cut-off value)
    p: float = 2 #'Norm associated to distance' (power value)
    

    def compute_cost_matrix(self, track_states, truth_states, complete=False):
        """Creates the cost matrix between two lists of states

        This distance measure here will return distances minimum of either
        'c' or the distance calculated from 'Measure'.

        Parameters
        ----------
        track_states: list of states
        truth_states: list of states
        complete: bool
            Cost matrix will be square, with 'c' present for where
            there is a mismatch in cardinality

        Returns
        -------
        cost_matrix: np.ndarray
            Matrix of distance between each element in each list of states
        """

        if complete:
            m = n = max((len(track_states), len(truth_states)))
        else:
            m, n = len(track_states), len(truth_states)

        # c could be int, so force to float
        cost_matrix = np.full((m, n), self.c, dtype=np.float64)

        for i_track, track_state, in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):
                if None in (track_state, truth_state):
                    continue

                distance_ = distance.euclidean(track_state, truth_state)

                if distance_ < self.c:
                    cost_matrix[i_track, i_truth] = distance_

        return cost_matrix

    def compute_OSPA_distance(self, track_states, truth_states):
        r"""
        Computes the Optimal SubPattern Assignment (OSPA) metric for a single
        time step between two point patterns. Each point pattern consisting of
        a list of states.

        The function :math:`\bar{d}_{p}^{(c)}` is the OSPA metric of order
        :math:`p` with cut-off :math:`c`. The OSPA metric is defined as:

            .. math::
                \begin{equation*}
                    \bar{d}_{p}^{(c)}({X},{Y}) :=
                    \Biggl( \frac{1}{n}
                    \Bigl({min}_{\substack{
                        \pi\in\Pi_{n}}}
                            \sum_{i=1}^{m}
                                d^{(c)}(x_{i},y_{\pi(i)})^{p}+
                                c^{p}(n-m)\Bigr)
                        \Biggr)^{ \frac{1}{p} }
                \end{equation*}

        Parameters
        ----------
        track_states: list of states
        truth_states: list of states

        Returns
        -------
        SingleTimeMetric
            The OSPA distance

        """

        if not track_states and not truth_states:
            distance = 0

        elif self.p < np.inf:
            cost_matrix = self.compute_cost_matrix(track_states, truth_states, complete=True)
            # Solve cost matrix with Hungarian/Munkres using
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Length of longest set of states
            n = max(len(track_states), len(truth_states))
            # Calculate metric
            distance = ((1/n) * np.sum(cost_matrix[row_ind, col_ind]**self.p))**(1/self.p)
        else:  # self.p == np.inf
            if len(track_states) == len(truth_states):
                cost_matrix = self.compute_cost_matrix(track_states, truth_states)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                distance = np.max(cost_matrix[row_ind, col_ind])
            else:
                distance = self.c

        return distance

"""
Example Usage:
"""
track_state = [(2.20, 1.), (2., 1), (23., 100)] #m

truth_state = [(4.2, 2)] # ground truth position

ospa_ = OSPAMetric()

ospa_distance = ospa_.compute_OSPA_distance(track_states=track_state, truth_states=truth_state)

print(ospa_distance)
