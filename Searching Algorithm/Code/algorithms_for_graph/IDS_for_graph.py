from graphProblem import *

class IDS_algorithm_for_graph:
    def __init__(self):
        self.algorithm_name = 'Iterative Deepening Search For Graph'

    def __call__(self, h = None):
        print ('Algorithm: {}'.format(self.algorithm_name))
        self.h = h
        return self.iterative_deepening_search_for_vis
        
    def depth_limited_search_graph(self, problem, limit = -1):
        '''
        Perform depth first search of graph g.
        if limit >= 0, that is the maximum depth of the search.
        '''
        # we use these two variables at the time of visualisations
        iterations = 0
        all_node_colors = []
        node_colors = {k : 'white' for k in problem.graph.nodes()}
        
        frontier = [Node(problem.initial)]
        explored = set()
        
        cutoff_occurred = False
        node_colors[Node(problem.initial).state] = "orange"
        iterations += 1
        all_node_colors.append(dict(node_colors))
          
        while frontier:
            # Popping first node of queue
            node = frontier.pop()
            
            # modify the currently searching node to red
            node_colors[node.state] = "red"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            
            if problem.goal_test(node.state):
                # modify goal node to green after reaching the goal
                node_colors[node.state] = "green"
                iterations += 1
                all_node_colors.append(dict(node_colors))
                return(iterations, all_node_colors, node)

            elif limit >= 0:
                cutoff_occurred = True
                limit += 1
                all_node_colors.pop()
                iterations -= 1
                node_colors[node.state] = "gray"

            
            explored.add(node.state)
            frontier.extend(child for child in node.expand(problem)
                            if child.state not in explored and
                            child not in frontier)
            
            for n in frontier:
                limit -= 1
                # modify the color of frontier nodes to orange
                node_colors[n.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))

            # modify the color of explored nodes to gray
            node_colors[node.state] = "gray"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            
        return 'cutoff' if cutoff_occurred else None


    def depth_limited_search_for_vis(self, problem):
        """Search the deepest nodes in the search tree first."""
        iterations, all_node_colors, node = self.depth_limited_search_graph(problem)
        return(iterations, all_node_colors, node) 

    def iterative_deepening_search_for_vis(self, problem):
        for depth in range(sys.maxsize):
            iterations, all_node_colors, node=self.depth_limited_search_for_vis(problem)
            if iterations:
                return (iterations, all_node_colors, node)