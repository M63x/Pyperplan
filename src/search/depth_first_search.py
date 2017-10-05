from . import searchspace

node_data = []
stack = []


def depth_first_search(planning_task):
    root = searchspace.make_root_node(planning_task.initial_state)
    stack.append(root)
    closed = {planning_task.initial_state}
    while len(stack) != 0:
        curr_node = stack.pop()
        node_data.append(curr_node)
        if planning_task.goal_reached(curr_node.state):
            return curr_node.extract_solution(), node_data
        for operator, successor_state in planning_task.get_successor_states(curr_node.state):
            if successor_state not in closed:
                closed.add(successor_state)
                stack.append(searchspace.make_child_node(curr_node, operator, successor_state))
    return None
