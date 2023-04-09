# This code is taken from https://towardsdatascience.com/solving-racetrack-in-reinforcement-learning-using-monte-carlo-control-bdee2aa4f04e
import numpy as np


class Generator:

    # HELPFUL FUNCTIONS
    def widen_hole_transformation(self, racetrack, start_cell, end_cell):

        δ = 1
        while (1):
            if ((start_cell[1] < δ) or (start_cell[0] < δ)):
                racetrack[0:end_cell[0], 0:end_cell[1]] = -1
                break

            if ((end_cell[1] + δ > 100) or (end_cell[0] + δ > 100)):
                racetrack[start_cell[0]:100, start_cell[1]:100] = -1
                break

            δ += 1

        return racetrack

    def calculate_valid_fraction(self, racetrack):
        '''
        Returns the fraction of valid cells in the racetrack
        '''
        return (len(racetrack[racetrack == 0]) / 10000)

    def mark_finish_states(self, racetrack):
        '''
        Marks finish states in the racetrack
        Returns racetrack
        '''
        last_col = racetrack[0:100, 99]
        last_col[last_col == 0] = 2
        return racetrack

    def mark_start_states(self, racetrack):
        '''
        Marks start states in the racetrack
        Returns racetrack
        '''
        last_row = racetrack[99, 0:100]
        last_row[last_row == 0] = 1
        return racetrack

    # CONSTRUCTOR
    def __init__(self):
        pass

    def generate_racetrack(self):
        '''
        racetrack is a 2d numpy array
        codes for racetrack:
            0,1,2 : valid racetrack cells
            -1: invalid racetrack cell
            1: start line cells
            2: finish line cells
        returns randomly generated racetrack
        '''
        racetrack = np.zeros((100, 100), dtype='int')

        frac = 1
        while frac > 0.5:
            # transformation
            random_cell = np.random.randint((100, 100))
            random_hole_dims = np.random.randint((25, 25))
            start_cell = np.array([max(0, x - y // 2) for x, y in zip(random_cell, random_hole_dims)])
            end_cell = np.array([min(100, x + y) for x, y in zip(start_cell, random_hole_dims)])

            # apply_transformation
            racetrack = self.widen_hole_transformation(racetrack, start_cell, end_cell)
            frac = self.calculate_valid_fraction(racetrack)

        racetrack = self.mark_start_states(racetrack)
        racetrack = self.mark_finish_states(racetrack)

        return racetrack

racetrack = Generator().generate_racetrack()
