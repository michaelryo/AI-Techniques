from graphProblem import *

class GBFS_algorithm_for_graph:

    def __init__(self):
        self.algorithm_name = 'GBFS_for_graph'

    def __call__(self, h = None):
        print ('Algorithm: {}'.format(self.algorithm_name))
        self.h = h
        return self.greedy_best_first_search_for_graph

    def best_first_graph_search_for_vis(self, problem, f):
        """Search the nodes with the lowest f scores first.
        You specify the function f(node) that you want to minimize; for example,
        if f is a heuristic estimate to the goal, then we have greedy best
        first search; if f is node.depth then we have breadth-first search.
        There is a subtlety: the line "f = memoize(f, 'f')" means that the f
        values will be cached on the nodes as they are computed. So after doing
        a best first search you can examine the f values of the path returned."""
        
        # we use these two variables at the time of visualisations
        iterations = 0
        all_node_colors = []
        node_colors = {k : 'white' for k in problem.graph.nodes()}

        node = Node(problem.initial)
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)
        
        frontier = PriorityQueue('min', f)
        frontier.append(node)
        
        node_colors[node.state] = "orange"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        explored = set()
        while frontier:
            node = frontier.pop()
            node_colors[node.state] = "red"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            
            if problem.goal_test(node.state):
                node_colors[node.state] = "green"
                iterations += 1
                all_node_colors.append(dict(node_colors))
                return(iterations, all_node_colors, node)
            explored.add(node.state)
            for child in node.expand(problem):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))
                elif child in frontier:
                    incumbent = frontier[child]
            node_colors[node.state] = "gray"
            iterations += 1
            all_node_colors.append(dict(node_colors))
        return None

    def greedy_best_first_search_for_graph(self, problem, h=None):
        """Greedy Best-first graph search is an informative searching algorithm with f(n) = h(n).
        You need to specify the h function when you call best_first_search, or
        else in your Problem subclass."""
        h = memoize(self.h or problem.h, 'h')
        iterations, all_node_colors, node = self.best_first_graph_search_for_vis(problem, lambda n: h(n))
        return(iterations, all_node_colors, node)