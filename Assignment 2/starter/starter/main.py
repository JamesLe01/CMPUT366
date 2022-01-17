import matplotlib.pyplot as plt
import numpy as np
import time

class PlotResults:
    """
    Class to plot the results. 
    """
    def plot_results(self, data1, data2, label1, label2, filename):
        """
        This method receives two lists of data point (data1 and data2) and plots
        a scatter plot with the information. The lists store statistics about individual search 
        problems such as the number of nodes a search algorithm needs to expand to solve the problem.

        The function assumes that data1 and data2 have the same size. 

        label1 and label2 are the labels of the axes of the scatter plot. 
        
        filename is the name of the file in which the plot will be saved.
        """
        _, ax = plt.subplots()
        ax.scatter(data1, data2, s=100, c="g", alpha=0.5, cmap=plt.cm.coolwarm, zorder=10)
    
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
    
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.grid()
        plt.savefig(filename)

class Grid:
    """
    Class to represent an assignment of values to the 81 variables defining a Sudoku puzzle. 

    Variable _cells stores a matrix with 81 entries, one for each variable in the puzzle. 
    Each entry of the matrix stores the domain of a variable. Initially, the domains of variables
    that need to have their values assigned are 123456789; the other domains are limited to the value
    initially assigned on the grid. Backtracking search and AC3 reduce the the domain of the variables 
    as they proceed with search and inference.
    """
    def __init__(self):
        self._cells = []
        self._complete_domain = "123456789"
        self._width = 9

    def copy(self):
        """
        Returns a copy of the grid. 
        """
        copy_grid = Grid()
        copy_grid._cells = [row.copy() for row in self._cells]
        return copy_grid

    def get_cells(self):
        """
        Returns the matrix with the domains of all variables in the puzzle.
        """
        return self._cells

    def get_width(self):
        """
        Returns the width of the grid.
        """
        return self._width

    def read_file(self, string_puzzle):
        """
        Reads a Sudoku puzzle from string and initializes the matrix _cells. 

        This is a valid input string:

        4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......

        This is translated into the following Sudoku grid:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        i = 0
        row = []
        for p in string_puzzle:
            if p == '.':
                row.append(self._complete_domain)
            else:
                row.append(p)

            i += 1

            if i % self._width == 0:
                self._cells.append(row)
                row = []
            
    def print(self):
        """
        Prints the grid on the screen. Example:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        for _ in range(self._width + 4):
            print('-', end=" ")
        print()

        for i in range(self._width):

            print('|', end=" ")

            for j in range(self._width):
                if len(self._cells[i][j]) == 1:
                    print(self._cells[i][j], end=" ")
                elif len(self._cells[i][j]) > 1:
                    print('.', end=" ")
                else:
                    print(';', end=" ")

                if (j + 1) % 3 == 0:
                    print('|', end=" ")
            print()

            if (i + 1) % 3 == 0:
                for _ in range(self._width + 4):
                    print('-', end=" ")
                print()
        print()

    def print_domains(self):
        """
        Print the domain of each variable for a given grid of the puzzle.
        """
        for row in self._cells:
            print(row)

    def is_solved(self):
        """
        Returns True if the puzzle is solved and False otherwise. 
        """
        for i in range(self._width):
            for j in range(self._width):
                if len(self._cells[i][j]) != 1:
                    return False
        return True

class VarSelector:
    """
    Interface for selecting variables in a partial assignment. 

    Extend this class when implementing a new heuristic for variable selection.
    """
    def select_variable(self, grid):
        pass

class FirstAvailable(VarSelector):
    """
    Naïve method for selecting variables; simply returns the first variable encountered whose domain is larger than one.
    """
    def select_variable(self, grid):
        # Implement here the first available heuristic

        # Sequentially loop all the cells until finding an unassigned variable
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(grid.get_cells()[i][j]) > 1:
                    return i, j
        return -1, -1  # All variables has been assigned, so return (-1, -1)

class MRV(VarSelector):
    """
    Implements the MRV heuristic, which returns one of the variables with smallest domain. 
    """
    def select_variable(self, grid):
        # Implement here the mrv heuristic
        minimum_remaining_value = float("inf")  # infinity
        minimum_remaining_value_position = (-1, -1)

        # Loop all the cells, find the one with minimum length
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                l = len(grid.get_cells()[i][j])
                if (l != 1) and (l < minimum_remaining_value):
                    minimum_remaining_value = len(grid.get_cells()[i][j])
                    minimum_remaining_value_position = (i, j)
        return minimum_remaining_value_position

class AC3:
    """
    This class implements the methods needed to run AC3 on Sudoku. 
    """
    def remove_domain_row(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same row. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != column:
                new_domain = grid.get_cells()[row][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[row][j]) > 1:
                    variables_assigned.append((row, j))

                grid.get_cells()[row][j] = new_domain
        
        return variables_assigned, False

    def remove_domain_column(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same column. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != row:
                new_domain = grid.get_cells()[j][column].replace(grid.get_cells()[row][column], '')
                
                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[j][column]) > 1:
                    variables_assigned.append((j, column))

                grid.get_cells()[j][column] = new_domain

        return variables_assigned, False

    def remove_domain_unit(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same unit. 
        """
        variables_assigned = []

        row_init = (row // 3) * 3
        column_init = (column // 3) * 3

        for i in range(row_init, row_init + 3):
            for j in range(column_init, column_init + 3):
                if i == row and j == column:
                    continue

                new_domain = grid.get_cells()[i][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[i][j]) > 1:
                    variables_assigned.append((i, j))

                grid.get_cells()[i][j] = new_domain
        return variables_assigned, False

    def pre_process_consistency(self, grid):
        """
        This method enforces arc consistency for the initial grid of the puzzle.

        The method runs AC3 for the arcs involving the variables whose values are 
        already assigned in the initial grid. 
        """
        # Implement here the code for making the CSP arc consistent as a pre-processing step; this method should be called once before search
        Q = []
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(grid.get_cells()[i][j]) == 1:
                    Q.append((i, j))
        temp = self.consistency(grid, Q)
        if (temp == "failure"):
            return None, "failure"
        return grid, "success"

    def consistency(self, grid, Q):
        """
        This is a domain-specific implementation of AC3 for Sudoku. 

        It keeps a set of variables to be processed (Q) which is provided as input to the method. 
        Since this is a domain-specific implementation, we don't need to maintain a graph and a set 
        of arcs in memory. We can store in Q the cells of the grid and, when processing a cell, we
        ensure arc consistency of all variables related to this cell by removing the value of
        cell from all variables in its column, row, and unit. 

        For example, if the method is used as a preprocessing step, then Q is initialized with 
        all cells that start with a number on the grid. This method ensures arc consistency by
        removing from the domain of all variables in the row, column, and unit the values of 
        the cells given as input. Like the general implementation of AC3, the method adds to 
        Q all variables that have their values assigned during the propagation of the contraints. 

        The method returns True if AC3 detected that the problem can't be solved with the current
        partial assignment; the method returns False otherwise. 
        """
        # Implement here the domain-dependent version of AC3.
        while len(Q) != 0:
            row, column = Q.pop()
            variables_assigned1, failure1 = self.remove_domain_row(grid, row, column)
            variables_assigned2, failure2 = self.remove_domain_column(grid, row, column)
            variables_assigned3, failure3 = self.remove_domain_unit(grid, row, column)
            if failure1 or failure2 or failure3:
                return "failure"
            
            variables_assigned = variables_assigned1 + variables_assigned2 + variables_assigned3
            for t in variables_assigned:
                if t not in Q:
                    Q.append(t)
        return "success"

class Backtracking:
    """
    Class that implements backtracking search for solving CSPs. 
    """

    def search(self, grid, var_selector):
        """
        Implements backtracking search with inference.
        """
        # Backtracking without inference

        # Checking whether the tree is complete or not
        complete = True
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(grid.get_cells()[i][j]) > 1:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            return grid
        
        row, col = var_selector.select_variable(grid)  # Variable to be assigned
        
        for d in grid.get_cells()[row][col]:
            if not self.consistent_check(grid, d, row, col):
                continue
            # If consistent:
            copy_grid = grid.copy()  # Copy the grid
            copy_grid.get_cells()[row][col] = d  # Assigned value to the copy version
            rb = self.search(copy_grid, var_selector)
            if rb != "failure":
                return rb
        return "failure"

    def search_AC3(self, grid, var_selector):
        # Backtracking with AC3 inference
        # Using the helper function to do all the work
        ac3 = AC3()
        ac3.pre_process_consistency(grid)  # Pre-process the grid before running Backtracking
        return self.helper_search_AC3(grid, var_selector)

    def helper_search_AC3(self, grid, var_selector):
        # Backtracking with AC3 inference

        # Checking whether the tree is complete or not
        complete = True
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(grid.get_cells()[i][j]) > 1:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            return grid

        row, col = var_selector.select_variable(grid)  # Variable to be assigned
        
        for d in grid.get_cells()[row][col]:
            if not self.consistent_check(grid, d, row, col):
                continue
            # If consistent:
            copy_grid = grid.copy()  # Copy the grid
            copy_grid.get_cells()[row][col] = d  # Assigned value to the copy version

            # Find unselected variables in copy_grid, then run AC3 on it
            ac3 = AC3()
            ri = ac3.consistency(copy_grid, [(row, col)])  # Run AC3 inference
            if (ri != "failure"):
                rb = self.helper_search_AC3(copy_grid, var_selector)
                if rb != "failure":
                    return rb
        
        return "failure"


    def consistent_check(self, grid, d, row, col):
        # Checking the consistency if we assign value d to position (row, col) in the matrix
        # Strategy: Check consistency in row and col, then check consistency in the unit

        # Check consistency within column and row
        for i in range(grid.get_width()):
            temp1 = grid.get_cells()[row][i]  # Iterate column
            temp2 = grid.get_cells()[i][col]  # Iterate row
            if len(temp1) == 1 and temp1 == d and i != col:
                return False
            if len(temp2) == 1 and temp2 == d and i != row:
                return False
        
        unit_row_position = (-1, -1)
        # unit_row_position is the range of row of the unit var is in
        # (-1, 1) is just a dummy value
        if (row % 3 == 0):
            unit_row_position = (row, row + 2)
        elif (row % 3 == 1):
            unit_row_position = (row - 1, row + 1)
        else:
            unit_row_position = (row - 2, row)
        
        unit_col_position = (-1, -1)
        # unit_col_position is the range of row of the unit var is in
        # (-1, 1) is just a dummy value
        if (col % 3 == 0):
            unit_col_position = (col, col + 2)
        elif (col % 3 == 1):
            unit_col_position = (col - 1, col + 1)
        else:
            unit_col_position = (col - 2, col)
        
        # Check consistency within a unit
        for i in range(unit_row_position[0], unit_row_position[1] + 1):
            for j in range(unit_col_position[0], unit_col_position[1] + 1):
                temp = grid.get_cells()[i][j]
                if len(temp) == 1 and temp == d and (i != row or j != col):
                    return False
        return True


file = open('top95.txt', 'r')
problems = file.readlines()

running_time_mrv = []
running_time_first_available = []
i = 1
for p in problems:
    print("\n\nSudoku", i)
    g = Grid()
    g.read_file(p)
    g.print()

    # MRV
    b = Backtracking()
    mrv = MRV()
    start = time.time()
    temp = b.search_AC3(g, mrv)
    end = time.time()
    running_time_mrv.append(end - start)
    temp.print()
    print("Is solved:", temp.is_solved())

    # FA
    fa = FirstAvailable()
    start = time.time()
    temp = b.search_AC3(g, fa)
    end = time.time()
    running_time_first_available.append(end - start)
    temp.print()
    print("Is solved:", temp.is_solved())
    
    i += 1

plotter = PlotResults()
plotter.plot_results(running_time_mrv, running_time_first_available, "Running Time Backtracking (MRV)", "Running Time Backtracking (FA)", "running_time")

