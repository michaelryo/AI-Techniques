from graphProblem import *

class AStar_algorithm_for_graph:
    def __init__(self):
        self.algorithm_name = 'AStar_for_graph'

    def __call__(self, h = None):
        print ('Algorithm: {}'.format(self.algorithm_name))
        self.h = h
        return self.astar_search_graph

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
        f = memoize(f, 'f')
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
                    if f(child) < frontier[child]:
                        del frontier[child]
                        frontier.append(child)
                        node_colors[child.state] = "orange"
                        iterations += 1
            node_colors[node.state] = "gray"
            iterations += 1
            all_node_colors.append(dict(node_colors))
        return None

    def astar_search_graph(self, problem, h=None):
        """A* search is best-first graph search with f(n) = g(n)+h(n).
        You need to specify the h function when you call astar_search, or
        else in your Problem subclass."""
        h = memoize(self.h or problem.h, 'h')
        return self.best_first_graph_search_for_vis(problem, lambda n: n.path_cost + h(n))
