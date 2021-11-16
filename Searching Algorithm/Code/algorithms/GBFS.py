from graphProblem import *

class GBFS_algorithm:

    def __init__(self):
        self.algorithm_name = 'GBFS'

    def __call__(self, h = None):
        print ('Algorithm: {}'.format(self.algorithm_name))
        self.h = h
        return self.greedy_best_first_search

    def best_first_graph_search(self, problem, f):
        """Search the nodes with the lowest f scores first.
        You specify the function f(node) that you want to minimize; for example,
        if f is a heuristic estimate to the goal, then we have greedy best
        first search; if f is node.depth then we have breadth-first search.
        There is a subtlety: the line "f = memoize(f, 'f')" means that the f
        values will be cached on the nodes as they are computed. So after doing
        a best first search you can examine the f values of the path returned."""
        
        # we use these two variables at the time of visualisations
        iterations = 0
        node = Node(problem.initial)
        iterations += 1
       
        if problem.goal_test(node.state):
            iterations += 1
            return(iterations, node)
        
        frontier = PriorityQueue('min', f)
        frontier.append(node)
        
        iterations += 1
        
        explored = set()
        while frontier:
            node = frontier.pop()
            iterations += 1
            if problem.goal_test(node.state):
                iterations += 1
                return(iterations, node)
            explored.add(node.state)
            for child in node.expand(problem):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                    iterations += 1
                elif child in frontier:
                    incumbent = frontier[child]
            iterations += 1
        return None

    def greedy_best_first_search(self, problem, h=None):
        """Greedy Best-first graph search is an informative searching algorithm with f(n) = h(n).
        You need to specify the h function when you call best_first_search, or
        else in your Problem subclass."""
        h = memoize(self.h or problem.h, 'h')
        iterations, node = self.best_first_graph_search(problem, lambda n: h(n))
        return(iterations, node)