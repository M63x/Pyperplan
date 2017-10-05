import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

depth = [0]
shown_nodes = list()
global_problem_parents = list()
shown_main_nodes = list()
global_dark_edges = list()
global_yellow_list = list()
global_green_list = list()
global_purple_list = list()
global_line_data = list()


class Shownnodes:
    index = 0
    pos = [0, 0]

    def __init__(self, index, pos):
        self.index = index
        self.pos = pos


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos == None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = G.neighbors(root)
    if parent != None:
        neighbors.remove(parent)
    if len(neighbors) != 0:
        dx = width/len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos


def take_dist(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**(1/2)

def visualization(nodes):
    def get_node_frame_coordinates():
        max_prev_w = -1000000
        min_prev_h = 1000000
        for elem in shown_nodes[:-1]:
            for element in elem:
                if element.pos[0] > max_prev_w:
                    max_prev_w = element.pos[0]
                if element.pos[1] < min_prev_h:
                    min_prev_h = element.pos[1]
        max_w = max_prev_w
        min_h = min_prev_h
        for elem in shown_nodes[-1]:
            if elem.pos[0] > max_w:
                max_w = elem.pos[0]
            if elem.pos[1] < min_h:
                min_h = elem.pos[1]
        if len(shown_nodes) == 1:
            max_prev_w = max_w
            min_prev_h = min_h
        return (max_w, max_prev_w, min_h, min_prev_h)

    def update_plot(is_add, arr):
        max_w, max_prev_w, min_h, min_prev_h = arr
        k = 100
        if is_add:
            k *= 1
        else:
            k *= -1
        g = 2
        ax = plt.gca()
        width_cnv = int(cnv.get_tk_widget()['width']) + k*(max_w-max_prev_w)
        height_cnv = int(cnv.get_tk_widget()['height']) + k*(min_prev_h-min_h)
        if width_cnv < 1000:
            width_cnv = 1000
        if height_cnv < 700:
            height_cnv = 700
        cnv.get_tk_widget().config(width=width_cnv,
                                   height=height_cnv)
        width_cnv = int(cnv.get_tk_widget()['width'])
        height_cnv = int(cnv.get_tk_widget()['height'])
        cnv.get_tk_widget().config(scrollregion=(0, 0, width_cnv*g-1010, height_cnv*g-700))

    class F_keeper:
        node = None

        def __init__(self, node):
            self.node = node

        def show(self):
            x_for_button = 0.875
            y_for_button = 0.825
            def command_more(event):
                more_button.place_forget()
                less_button.place(relx=x_for_button, rely=y_for_button)
                additional_viz(nodes.index(node))

            def command_less(event):
                less_button.place_forget()
                more_button.place(relx=x_for_button, rely=y_for_button)
                additional_invis(nodes.index(node))

            node = self.node

            more_button.place_forget()
            less_button.place_forget()

            inform = []
            inform.append("Information about " + str(nodes.index(node)) + "-node")
            if node.parent:
                inform.append("Depth: "+str(node.g))
            else:
                inform.append("It is the root node")
            inform.append("State of node:")
            for elem in node.state:
                inform.append(elem)
            if node.action:
                inform.append("List of actions:")
                for elem in (re.split(r"  ", str(node.action))):
                    inform.append(elem)
            else:
                inform.append("There is not any changes")
            for elem in global_problem_parents:
                if nodes.index(node) in elem:
                    more_button.bind('<Button-1>', command_more)
                    less_button.bind('<Button-1>', command_less)
                    if nodes.index(node) not in shown_main_nodes:
                        more_button.place(relx=x_for_button, rely=y_for_button)
                    else:
                        less_button.place(relx=x_for_button, rely=y_for_button)
                    break
            else:
                more_button.place_forget()
                less_button.place_forget()

            nonlocal listbox_for_inform
            listbox_for_inform.pack_forget()
            listbox_for_inform = tk.Listbox(frame_for_inform, height=400, width=200)
            for i, elem in enumerate(inform):
                listbox_for_inform.insert(tk.END, elem)
            listbox_for_inform.pack()
            scrolly.pack(side=tk.RIGHT, fill=tk.Y)
            scrollx.pack(side=tk.BOTTOM, fill=tk.X)
            listbox_for_inform.config(yscrollcommand=scrolly.set)
            listbox_for_inform.config(xscrollcommand=scrollx.set)
            scrolly.config(command=listbox_for_inform.yview)
            scrollx.config(command=listbox_for_inform.xview)

    def onclick(event):
        try:
            x = cnv.get_tk_widget().canvasx(event.x)
            y = cnv.get_tk_widget().canvasy(event.y)
            x, y = event.inaxes.transData.inverted().transform((x, y))
            diff_y = y - event.ydata
        except BaseException:
            return None
        y -= 2*diff_y
        for elem in shown_nodes:
            for element in elem:
                dist = ((x - element.pos[0]) ** 2 + (y - element.pos[1]) ** 2) ** (1 / 2)
                if dist < 0.015:
                    list_of_keepers[element.index].show()

    def additional_invis(node):
        def del_sons(i, list_of_sons):
            list_of_shown_nodes_i = list()
            for element in shown_nodes[i]:
                list_of_shown_nodes_i.append(element.index)
            for j, element in enumerate(list_of_shown_nodes_i):
                if element in global_purple_list[i]:
                    for k in range(i + 1, len(shown_nodes)):
                        list_of_shown_nodes_k = list()
                        for keke in shown_nodes[k]:
                            list_of_shown_nodes_k.append(keke.index)
                        if element in list_of_shown_nodes_k:
                            del_sons(k, list_of_sons)
            list_of_sons.append(i)

        plt.clf()
        coords = get_node_frame_coordinates()
        for i, elem in enumerate(shown_main_nodes):
            if elem == node:
                list_of_sons = list()
                del_sons(i, list_of_sons)
                list_of_sons.sort()
                list_of_sons.reverse()
                for ind in list_of_sons:
                    diff = 0
                    if ind+1 < len(shown_nodes):
                        diff = depth[ind+1] - depth[ind]
                    for k in range(ind, len(depth)):
                        depth[k] -= diff
                        for element in shown_nodes[k]:
                            element.pos = (element.pos[0], element.pos[1] - diff)
                    for k in range(ind, len(global_line_data)):
                        check = False
                        kuk = False
                        for element in reversed(shown_nodes):
                            for keken in element:
                                try:
                                    if keken.index == shown_main_nodes[k + 1]:
                                        if kuk:
                                            find = keken
                                            check = True
                                        kuk = True
                                        break
                                except BaseException:
                                    break
                            if check:
                                break
                        global_line_data[k][1] = find.pos[1]
                    del shown_main_nodes[ind]
                    del shown_nodes[ind]
                    del global_problem_parents[ind]
                    del global_dark_edges[ind]
                    del global_yellow_list[ind]
                    del global_green_list[ind]
                    del global_purple_list[ind]
                    del global_line_data[ind - 1]
                    del depth[ind]

        for i, element in enumerate(shown_main_nodes):
            shown_main_list = list()
            shown_main_pos = dict()
            for elem in tree.nodes():
                shown_main_pos[elem] = (0, 0)
            for elem in shown_nodes[i]:
                shown_main_pos[elem.index] = elem.pos
                shown_main_list.append(elem.index)

            nx.draw_networkx_nodes(tree, pos=shown_main_pos, nodelist=shown_main_list, node_size=100)
            nx.draw_networkx_edges(tree, pos=shown_main_pos, edgelist=global_dark_edges[i])
            nx.draw_networkx_nodes(tree, pos=shown_main_pos, nodelist=global_yellow_list[i],
                                   node_color='yellow', node_size=100)
            nx.draw_networkx_nodes(tree, pos=shown_main_pos, nodelist=global_purple_list[i],
                                   node_color='purple', node_size=100)
            nx.draw_networkx_nodes(tree, pos=shown_main_pos, nodelist=global_green_list[i],
                                   node_color='green', node_size=100)
            if i != 0:
                plt.plot([global_line_data[i - 1][0],
                          global_line_data[i - 1][0]],
                         [global_line_data[i - 1][1],
                          depth[i]], linewidth=2)
            plt.axis('off')
        fig.canvas.draw()
        update_plot(False, coords)

    def show_solution(event):
        while True:
            length = 1
            curr_elem = -1
            for i, elem in enumerate(global_green_list):
                if len(elem) != 0:
                    curr_elem = elem[0]
                    length = i
            if curr_elem not in shown_main_nodes and curr_elem != -1:
                additional_viz(curr_elem)
            new_length = 1
            for i, elem in enumerate(global_green_list):
                if len(elem) != 0:
                    new_length = i
            if length == new_length:
                break

    def reset(event):
        for elem in reversed(shown_main_nodes):
            if elem != 0:
                additional_invis(elem)
        g = 2
        width_cnv = 1000
        height_cnv = 700
        cnv.get_tk_widget().config(width=width_cnv, height=height_cnv)
        width_cnv = int(cnv.get_tk_widget()['width'])
        height_cnv = int(cnv.get_tk_widget()['height'])
        cnv.get_tk_widget().config(scrollregion=(0, 0, width_cnv*g-1010, height_cnv*g-700))
        less_button.place_forget()

    def additional_viz(node):
        def get_local_dark_edges_and_nodes():
            """Create three different sets with information about
            edges and nodes which should be disappeared"""

            def removing_nodes(main_node):
                for elem in tree.neighbors(main_node):
                    if nodes[elem].parent:
                        if nodes.index(nodes[elem].parent) == main_node:
                            removing_nodes(elem)
                            dark_set_edges.add((main_node, elem))
                dark_set_nodes.add(main_node)

            set_parent_problem = set()
            dark_set_nodes = set()
            dark_set_edges = set()
            for elem in list_of_undernodes:
                x_min = 100000000
                x_max = -100000000
                try:
                    parent_index = nodes.index(nodes[elem].parent)
                except ValueError:
                    parent_index = None
                for neighbour in tree.neighbors(elem):
                    if neighbour != parent_index:
                        if x_min > local_pos[neighbour][0]:
                            x_min = local_pos[neighbour][0]
                        if x_max < local_pos[neighbour][0]:
                            x_max = local_pos[neighbour][0]
                if len(tree.neighbors(elem)) > 2:
                    if abs(x_max - x_min) / (len(tree.neighbors(elem)) - 1) < 0.018:
                        set_parent_problem.add(elem)
                        for neighbour in tree.neighbors(elem):
                            if nodes[elem].parent:
                                if nodes.index(nodes[neighbour].parent) == elem:
                                    removing_nodes(neighbour)
            for elem in set_parent_problem:
                for element in tree.neighbors(elem):
                    if nodes.index(nodes[element].parent) == elem:
                        dark_set_edges.add((elem, nodes.index(nodes[element])))
            return set_parent_problem, dark_set_nodes, dark_set_edges

        def _find_everyone(node):
            for neighbor in tree.neighbors(node):
                if nodes[neighbor].parent:
                    if nodes.index(nodes[neighbor].parent) == node and neighbor not in list_of_undernodes:
                        list_of_undernodes.add(neighbor)
                        list_of_underedges.add((node, neighbor))
                        _find_everyone(neighbor)

        list_of_undernodes = set()
        list_of_underedges = set()
        list_of_undernodes.add(node)
        _find_everyone(node)
        max_depth = 0
        for elem in shown_nodes:
            for element in elem:
                if element.pos[1] < max_depth:
                    max_depth = element.pos[1]
        if node != 0:
            depth.append(max_depth-0.2)
        horiz, vert = (0, 0)
        for elem in shown_nodes:
            for element in elem:
                if element.index == node:
                    horiz, vert = element.pos
                    break
        local_pos = hierarchy_pos(tree, node, vert_loc=depth[-1], xcenter=horiz)
        set_parent_local_problem, local_dark_set_nodes, local_dark_set_edges = get_local_dark_edges_and_nodes()
        global_problem_parents.append(set_parent_local_problem)
        new_nodes = list()
        for elem in list_of_undernodes-local_dark_set_nodes:
            new_nodes.append(Shownnodes(elem, local_pos[elem]))
        shown_nodes.append(new_nodes)
        shown_main_nodes.append(node)
        global_dark_edges.append((set(list_of_underedges)-local_dark_set_edges))
        addit_solution_list = set()
        green_list = set()
        check = True
        for elem in shown_nodes[-1]:
            if elem.index in solution and (elem.index not in local_dark_set_nodes):
                addit_solution_list.add(elem.index)
                if elem.index in set_parent_local_problem:
                    green_list.add(elem.index)
                    global_green_list.append([elem.index])
                    check = False
                    break
        if check:
            global_green_list.append([])
        global_yellow_list.append(addit_solution_list)
        global_purple_list.append(set_parent_local_problem-local_dark_set_nodes)
        if node != 0:
            global_line_data.append([horiz, vert])
        nx.draw_networkx(tree, local_pos, nodelist=(set(list_of_undernodes)-local_dark_set_nodes),
                edgelist=(set(list_of_underedges)-local_dark_set_edges), with_labels=False, node_size=100)
        nx.draw_networkx_nodes(tree, local_pos,
                               nodelist=addit_solution_list, node_color='yellow', node_size=100)
        nx.draw_networkx_nodes(tree, pos=local_pos, nodelist=(set_parent_local_problem-local_dark_set_nodes),
                               node_color='purple', node_size=100)
        nx.draw_networkx_nodes(tree, pos=local_pos, nodelist=green_list,
                               node_color='green', node_size=100)
        plt.plot([horiz, horiz], [vert, depth[-1]], linewidth=2)
        plt.axis("off")
        fig.canvas.draw()
        coords = get_node_frame_coordinates()
        update_plot(True, coords)

    solution = []
    current_node = nodes[-1]
    while current_node.parent:
        solution.append(nodes.index(current_node))
        current_node = current_node.parent
    solution.append(nodes.index(current_node))
    solution.reverse()
    matrix = np.matrix([[False]*len(nodes)]*len(nodes))
    for i, elem in enumerate(nodes):
        if elem.parent:
            matrix[i, nodes.index(elem.parent)] = True
            matrix[nodes.index(elem.parent), i] = True
    tree = nx.Graph(matrix)
    window = tk.Tk()
    window.geometry("800x600")
    window.title("Graph visualization")

    frame_for_plot = tk.Frame(window)
    frame_for_plot.place(relx=0, rely=0, relheight=1, relwidth=0.8)
    fig = plt.figure()
    cnv = FigureCanvasTkAgg(fig, master=frame_for_plot)
    hbar = tk.Scrollbar(frame_for_plot, orient=tk.HORIZONTAL)
    vbar = tk.Scrollbar(frame_for_plot, orient=tk.VERTICAL)

    cnv.mpl_connect('button_press_event', onclick)
    cnv.get_tk_widget().config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    hbar.config(command=cnv.get_tk_widget().xview)
    vbar.config(command=cnv.get_tk_widget().yview)
    cnv.get_tk_widget().place(relx=0, rely=0)
    vbar.place(x=1000, y=0, relheight=1)
    hbar.place(x=0, y=640, relwidth=0.977)
    cnv.get_tk_widget().config(width=1010, height=700)

    frame_for_inform = tk.LabelFrame(window, width=200, height=400)
    frame_for_inform.place(relx=0.825, rely=0.2)
    frame_for_inform.pack_propagate(0)

    listbox_for_inform = tk.Listbox(frame_for_inform, height=400, width=200)
    listbox_for_inform.pack()
    scrolly = tk.Scrollbar(frame_for_inform)
    scrolly.pack(side=tk.RIGHT, fill=tk.Y)
    scrollx = tk.Scrollbar(frame_for_inform, orient=tk.HORIZONTAL)
    scrollx.pack(side=tk.BOTTOM, fill=tk.X)
    listbox_for_inform.config(yscrollcommand=scrolly.set)
    listbox_for_inform.config(xscrollcommand=scrollx.set)
    scrolly.config(command=listbox_for_inform.yview)
    scrollx.config(command=listbox_for_inform.xview)

    more_button = tk.Button(window, text="Show\nmore")
    less_button = tk.Button(window, text="Show\nless")

    button_for_solution = tk.Button(window, text="Show\nsolution")
    button_for_solution.bind('<Button-1>', show_solution)
    button_for_solution.place(relx=0.9, rely=0.9)

    button_for_reset = tk.Button(window, text="Reset\nthe graph")
    button_for_reset.bind('<Button-1>', reset)
    button_for_reset.place(relx=0.85, rely=0.9)

    frame_for_legend = tk.LabelFrame(window, width=200, height=100)
    frame_for_legend.place(relx=0.825, rely=0.02)
    frame_for_legend.pack_propagate(0)
    canvas_for_legend = tk.Canvas(frame_for_legend, width=200, height=100, bg="white")
    canvas_for_legend.pack(expand=tk.YES, fill=tk.BOTH)
    canvas_for_legend.create_oval(10, 10, 30, 30, fill="red")
    canvas_for_legend.create_oval(10, 30, 30, 50, fill="yellow")
    canvas_for_legend.create_oval(10, 50, 30, 70, fill="purple")
    canvas_for_legend.create_oval(10, 70, 30, 90, fill="green")
    canvas_for_legend.create_text(72, 20, text='- Simple node')
    canvas_for_legend.create_text(76, 40, text='- Solution node')
    canvas_for_legend.create_text(93, 60, text='- Hiding simple nodes')
    canvas_for_legend.create_text(97, 80, text='- Hiding solution nodes')

    additional_viz(0)

    list_of_keepers = list()
    for elem in nodes:
        list_of_keepers.append(F_keeper(elem))

    window.protocol("WM_DELETE_WINDOW", lambda: exit(0))
    window.mainloop()
