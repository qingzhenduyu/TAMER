# from: https://github.com/yqingli123/TDv2/blob/main/utils/latex2gtd_v2_2.py

import random

import numpy as np
from dataclasses import dataclass


@dataclass
class Symbol:
    idx: int
    token: str

    def __eq__(self, __value: object) -> bool:
        assert isinstance(__value, str)
        return self.token == __value


class Node:
    def __init__(self, x=0):
        self.x = x
        self.childs = []
        self.relations = []


def findnextbracket(latex: list, leftbracket="{"):
    if leftbracket == "{":
        rightbracket = "}"
    elif leftbracket == "[":
        rightbracket = "]"
    else:
        raise AssertionError("Unkown Bracket!")

    num = 0
    for li, l in enumerate(latex):
        if l == leftbracket:
            num += 1
        if l == rightbracket:
            num -= 1
            if num == 0:
                return li
    return -1


def findendmatrix(latex):
    num = 1
    for li, l in enumerate(latex):
        if l == "\\begin{matrix}":
            num += 1
        if l == "\\end{matrix}":
            num -= 1
            if num == 0:
                return li
    return -1


def latex2Tree(latex: list):
    """
    input:  latex --> list
    output: Node
    """
    if len(latex) == 0:
        return Node("<eol>")

    cur_node = Node(latex[0])
    symbol = latex.pop(0)

    if symbol == "<bol>":
        if len(latex) > 0 and latex[0] == "_":
            latex.pop(0)
            assert latex[0] == "{", "_ not with {"
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("sub")
            for _ in range(li + 1):
                latex.pop(0)
        if len(latex) > 0 and latex[0] == "^":
            latex.pop(0)
            assert latex[0] == "{", "^ not with {"
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("sup")
            for _ in range(li + 1):
                latex.pop(0)
            li = findnextbracket(latex, leftbracket="{")

    elif symbol == "\\begin{matrix}":
        li = findendmatrix(latex)
        sub_latex = latex[:li]
        node = latex2Tree(sub_latex)
        cur_node.childs.append(node)
        cur_node.relations.append("Mstart")
        for _ in range(li + 1):
            latex.pop(0)

    elif symbol in ["\\iint", "\\bigcup", "\\sum", "\\lim", "\\coprod"]:
        if len(latex) > 0 and latex[0] == "_":
            latex.pop(0)
            assert latex[0] == "{", "_ not with {"
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("below")
            for _ in range(li + 1):
                latex.pop(0)
        if len(latex) > 0 and latex[0] == "^":
            latex.pop(0)
            assert latex[0] == "{", "^ not with {"
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("above")
            for _ in range(li + 1):
                latex.pop(0)

    elif symbol in [
        "\\dot",
        "\\ddot",
        "\\hat",
        "\\check",
        "\\grave",
        "\\acute",
        "\\tilde",
        "\\breve",
        "\\bar",
        "\\vec",
        "\\widehat",
        "\\overbrace",
        "\\widetilde",
        "\\overleftarrow",
        "\\overrightarrow",
        "\\overline",
    ]:
        assert latex[0] == "{", "CASE 3 above not with {"
        li = findnextbracket(latex, leftbracket="{")
        sub_latex = latex[1:li]
        node = latex2Tree(sub_latex)
        cur_node.childs.append(node)
        cur_node.relations.append("below")
        for _ in range(li + 1):
            latex.pop(0)

    elif symbol in ["\\underline", "\\underbrace"]:
        assert latex[0] == "{", "CASE 3 above not with {"
        li = findnextbracket(latex, leftbracket="{")
        sub_latex = latex[1:li]
        node = latex2Tree(sub_latex)
        cur_node.childs.append(node)
        cur_node.relations.append("above")
        for _ in range(li + 1):
            latex.pop(0)

    elif symbol in ["\\xrightarrow", "\\xleftarrow"]:
        if latex[0] == "[":
            li = findnextbracket(latex, leftbracket="[")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("below")
            for _ in range(li + 1):
                latex.pop(0)
        if latex[0] == "{":
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("above")
            for _ in range(li + 1):
                latex.pop(0)

    elif symbol == "\\frac":
        assert latex[0] == "{", "\\frac above not with {"
        li = findnextbracket(latex, leftbracket="{")
        sub_latex = latex[1:li]
        node = latex2Tree(sub_latex)
        cur_node.childs.append(node)
        cur_node.relations.append("above")
        for _ in range(li + 1):
            latex.pop(0)
        assert latex[0] == "{", "\\frac below not with {"
        li = findnextbracket(latex, leftbracket="{")
        sub_latex = latex[1:li]
        node = latex2Tree(sub_latex)
        cur_node.childs.insert(-1, node)
        cur_node.relations.insert(-1, "below")
        for _ in range(li + 1):
            latex.pop(0)

    elif symbol == "\\sqrt":
        if latex[0] == "[":
            li = findnextbracket(latex, leftbracket="[")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("leftup")
            for _ in range(li + 1):
                latex.pop(0)
        assert latex[0] == "{", "\\sqrt inside not with {"
        li = findnextbracket(latex, leftbracket="{")
        sub_latex = latex[1:li]
        node = latex2Tree(sub_latex)
        cur_node.childs.append(node)
        cur_node.relations.append("inside")
        for _ in range(li + 1):
            latex.pop(0)

    else:
        if len(latex) > 0 and latex[0] == "_":
            latex.pop(0)
            assert latex[0] == "{", "_ not with {"
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("sub")
            for _ in range(li + 1):
                latex.pop(0)
        if len(latex) > 0 and latex[0] == "^":
            latex.pop(0)
            assert latex[0] == "{", "^ not with {"
            li = findnextbracket(latex, leftbracket="{")
            sub_latex = latex[1:li]
            node = latex2Tree(sub_latex)
            cur_node.childs.append(node)
            cur_node.relations.append("sup")
            for _ in range(li + 1):
                latex.pop(0)

    if len(latex) > 0 and latex[0] == "\\\\":
        latex.pop(0)
        relation = "nextline"
    elif len(latex) > 0:
        relation = "right"
    else:
        relation = "end"
    node = latex2Tree(latex)
    cur_node.childs.append(node)
    cur_node.relations.append(relation)

    return cur_node


# index = 0
# def node2list(parent, parent_index, relation, current, gtd, initial=None):
#     if current is None or current.x == '<eol>':
#         return
#     global index
#     if initial is not None:
#         index = 1
#     else:
#         index = index + 1
#     gtd.append([current.x, index, parent, parent_index, relation])
#     parent_index = index
#     for child, relation in zip(current.childs, current.relations):
#         node2list(current.x, parent_index, relation, child, gtd)

# def node2list_shuffle(parent, parent_index, relation, current, gtd, initial=None):
#     if current is None:
#         return
#     global index
#     if initial is not None:
#         index = 1
#     else:
#         index = index + 1
#     gtd.append([current.x, index, parent, parent_index, relation])
#     parent_index = index
#     zip_childs = list(zip(current.childs, current.relations))
#     random.shuffle(zip_childs)
#     for child, relation in zip_childs:
#         node2list_shuffle(current.x, parent_index, relation, child, gtd)


def node2list(tree: Node, is_shuffle: bool = False):
    index = 0
    gtd = []

    def _node2list(parent, parent_index, relation, current, initial=None):
        if current is None or current.x == "<eol>":
            return
        nonlocal index
        if initial is not None:
            index = 1
        else:
            index = index + 1
        gtd.append([current.x, index, parent, parent_index, relation])
        parent_index = index
        zip_childs = list(zip(current.childs, current.relations))
        if is_shuffle:
            random.shuffle(zip_childs)
        for child, relation in zip_childs:
            _node2list(current.x, parent_index, relation, child)

    _node2list(Symbol(-1, "<sos>"), 0, "start", tree, initial=True)
    return gtd


def list2node(gtd):
    node_list = []
    root = Node("root")
    node_list.append(root)
    for g in gtd:
        child_node = Node(g[0])
        node_list.append(child_node)
        parent_node = node_list[g[3]]
        parent_node.childs.append(child_node)
        parent_node.relations.append(g[4])
    return node_list[1]


def tree2latex(root):
    symbol = root.x
    latex = [symbol]
    #### sort childs
    if len(root.childs) > 0:
        prios = [
            "start",
            "sub",
            "sup",
            "below",
            "above",
            "leftup",
            "inside",
            "right",
            "pad",
        ]
        tups = list(zip(root.childs, root.relations))
        tups.sort(key=lambda t: prios.index(t[1]))
        root.childs, root.relations = zip(*tups)
    ####
    if symbol == "<eol>":
        return []
    elif symbol == "\\begin{matrix}":
        for child, relation in zip(root.childs, root.relations):
            if relation == "Mstart":
                latex += tree2latex(child)
                latex.append("\\end{matrix}")
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)

    elif symbol == "\\frac":
        below_latex = []
        for child, relation in zip(root.childs, root.relations):
            if relation == "below":
                below_latex.append("{")
                below_latex += tree2latex(child)
                below_latex.append("}")
            elif relation == "above":
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
                latex += below_latex
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)
    elif symbol in [
        "\\frac",
        "\\underline",
        "\\underbrace",
        "\\dot",
        "\\ddot",
        "\\hat",
        "\\check",
        "\\grave",
        "\\acute",
        "\\tilde",
        "\\breve",
        "\\bar",
        "\\vec",
        "\\widehat",
        "\\overbrace",
        "\\widetilde",
        "\\overleftarrow",
        "\\overrightarrow",
        "\\overline",
    ]:
        for child, relation in zip(root.childs, root.relations):
            if relation in ["above", "below"]:
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)
    elif symbol == "\\sqrt":
        for child, relation in zip(root.childs, root.relations):
            if relation == "leftup":
                latex.append("[")
                latex += tree2latex(child)
                latex.append("]")
            elif relation == "inside":
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)
    elif symbol in ["\\xrightarrow", "\\xleftarrow"]:
        for child, relation in zip(root.childs, root.relations):
            if relation == "below":
                latex.append("[")
                latex += tree2latex(child)
                latex.append("]")
            elif relation == "above":
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)
    elif symbol in ["\\iint", "\\bigcup", "\\sum", "\\lim", "\\coprod"]:
        for child, relation in zip(root.childs, root.relations):
            if relation == "below":
                latex.append("_")
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "above":
                latex.append("^")
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)
    else:
        for child, relation in zip(root.childs, root.relations):
            if relation == "sub":
                latex.append("_")
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "sup":
                latex.append("^")
                latex.append("{")
                latex += tree2latex(child)
                latex.append("}")
            elif relation == "nextline":
                latex.append("\\\\")
                latex += tree2latex(child)
            else:
                latex += tree2latex(child)
    return latex


def relation2gtd(objects, relations, id2object, id2relation):
    gtd = [[] for o in objects]
    num_relation = len(relations[0])

    start_relation = np.array([0 for _ in range(num_relation)])
    start_relation[0] = 1
    relation_stack = [start_relation]
    parent_stack = [(len(id2object) - 1, 0)]
    p_re = len(id2relation) - 1
    p_y = len(id2object) - 1
    p_id = 0

    for ci, c in enumerate(objects):
        gtd[ci].append(id2object[c])
        gtd[ci].append(ci + 1)

        find_flag = False
        while relation_stack != []:
            if relation_stack[-1][: num_relation - 1].sum() > 0:
                for index_relation in range(num_relation):
                    if relation_stack[-1][index_relation] != 0:
                        p_re = index_relation
                        p_y, p_id = parent_stack[-1]
                        relation_stack[-1][index_relation] = 0
                        if relation_stack[-1][: num_relation - 1].sum() == 0:
                            relation_stack.pop()
                            parent_stack.pop()
                        find_flag = 1
                        break
            else:
                relation_stack.pop()
                parent_stack.pop()

            if find_flag:
                break

        if not find_flag:
            p_y = objects[ci - 1]
            p_id = ci
            p_re = num_relation - 1
        gtd[ci].append(id2object[p_y])
        gtd[ci].append(p_id)
        gtd[ci].append(id2relation[p_re])

        relation_stack.append(relations[ci])
        parent_stack.append((c, ci + 1))

    return gtd


def to_struct(latex_list):
    symbols = [Symbol(i, t) for i, t in enumerate(latex_list)]
    tree = latex2Tree(symbols)
    gtd = node2list(tree)
    d = {g[0].idx: g[2].idx for g in gtd}
    ret = [d[i] if i in d else -1 for i in range(len(latex_list))]
    return ret

def tree_complexity(latex: str):
    latex_ls = [Symbol(i, t) for i, t in enumerate(latex.split())]
    tree = latex2Tree(latex_ls)

    def complexity(node):
        child_cplx = 0
        for c in node.childs:
            child_cplx = max(child_cplx, complexity(c))
        curr_cplx = 1 if len(node.childs) > 1 else 0
        return curr_cplx + child_cplx

    return complexity(tree)


if __name__ == "__main__":
    # latex = "a = \\frac { x } { y } + \sqrt [ c ] { b }"
    latex = r"( \frac { 1 + x ^ { 2 } } { 1 + y ^ { 2 } } ) ^ { t } \leq 2 ^ { | t | } ( 1 + ( x - y ) ^ { 2 } ) ^ { | t | }"
    latex_ls = [Symbol(i, t) for i, t in enumerate(latex.split())]
    tree = latex2Tree(latex_ls)
    gtd = node2list(tree)
    print("original gtd:")
    for g in gtd:
        g = [str(item) for item in g]
        print("\t\t".join(g))
    print(to_struct(latex.split()))
    print(tree_complexity(r"\sum _ { 1 } \frac { x ^ { 2 } + 1 } { y }"))
    # predict_tree = list2node(gtd)
    # predict_latex = tree2latex(predict_tree)
    # predict_latex = " ".join(predict_latex)

    # print("shuffled gtd:")
    # gtd = node2list(tree, is_shuffle=True)
    # for g in gtd:
    #     g = [str(item) for item in g]
    #     print("\t\t".join(g))
    # shuffle_tree = list2node(gtd)
    # shuffle_latex = tree2latex(shuffle_tree)
    # shuffle_latex = " ".join(shuffle_latex)
    # print(latex == predict_latex)
    # print(latex)
    # print(predict_latex)
    # print(shuffle_latex)
