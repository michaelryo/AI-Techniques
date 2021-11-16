from graphProblem import *

class IDS_algorithm:
    def __init__(self):
        self.algorithm_name = 'Iterative Deepening Search'

    def __call__(self, h = None):
        print ('Algorithm: {}'.format(self.algorithm_name))
        self.h = h
        return self.iterative_deepening_search
        
    def depth_limited_search(self, problem, limit = -1):
        '''
        Perform depth first search of graph g.
        if limit >= 0, that is the maximum depth of the search.
        '''
        # we use these two variables at the time of visualisations
        iterations = 0
        
        frontier = [Node(problem.initial)]
        explored = set()
        
        cutoff_occurred = False
        iterations += 1
          
        while frontier:
            # Popping first node of queue
            node = frontier.pop()
            iterations += 1
            
            if problem.goal_test(node.state):
                iterations += 1
                return(iterations, node)

            elif limit >= 0:
                cutoff_occurred = True
                limit += 1
                iterations -= 1

            
            explored.add(node.state)
            frontier.extend(child for child in node.expand(problem)
                            if child.state not in explored and
                            child not in frontier)
            
            for n in frontier:
                limit -= 1
                iterations += 1
            iterations += 1
            
        return 'cutoff' if cutoff_occurred else None


    def iterative_deepening_search(self, problem):
        for depth in range(sys.maxsize):
            iterations, node=self.depth_limited_search(problem)
            if iterations:
                return (iterations, node)