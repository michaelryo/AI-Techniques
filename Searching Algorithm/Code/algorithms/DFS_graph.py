from graphProblem import *

class DFS_graph_algorithm:

    def __init__(self):
        self.algorithm_name = 'DFS_graph'

    def __call__(self):
        print ('Algorithm: {}'.format(self.algorithm_name))
        return self.depth_first_graph_search

    def depth_first_graph_search(self,problem):
        """
        [Figure 3.7]
        Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Does not get trapped by loops.
        If two paths reach a state, only use the first one.
        """
        iterations = 0
        frontier = [Node(problem.initial)]  # Stack
        iterations += 1
        explored = set()
        while frontier:
            node = frontier.pop()
            iterations += 1
            if problem.goal_test(node.state):
                iterations += 1
                return (iterations, node)
            explored.add(node.state)
            frontier.extend(child for child in node.expand(problem) if child.state not in explored and
                            child not in frontier)
            for n in frontier:
                iterations += 1
            iterations += 1
        return None