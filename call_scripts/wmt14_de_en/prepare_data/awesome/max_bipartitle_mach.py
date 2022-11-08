# coding=utf-8

import os
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import bipartite
import argparse
import torch
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
# maximal Bipartite matching.
# python program to find

def read_data(data_type, root_path, src_lang, tgt_lang):
    align_data_path = root_path + data_type + ".align.de-en"
    with open(align_data_path, encoding="utf-8") as f:
        align_lines = f.readlines()
    
    src_data_path = root_path + data_type + "." + src_lang
    with open(src_data_path, encoding="utf-8") as f:
        src_lines = f.readlines()

    tgt_data_path = root_path + data_type + "." + tgt_lang
    with open(tgt_data_path, encoding="utf-8") as f:
        tgt_lines = f.readlines()
    
    return align_lines, src_lines, tgt_lines


def reorder(graph,src_line, len_src, len_tgt, components):
    src_data_reordered = src_line.copy()
    src_idx_remain = list(range(len_src))
    if len_src >= len_tgt :
        src_idx_reordered = src_idx_remain.copy()
        src_replace_table = [False]*len_src
    else:
        src_idx_reordered = list(range(len_tgt)) 
        src_replace_table = [False]*len_tgt
    for com_nodes in components :
        graph_sub = graph.subgraph(com_nodes)
        left_nodes, right_nodes = nx.bipartite.sets(graph_sub)
        left_nodes = list(left_nodes)
        right_nodes = list(right_nodes)
        if left_nodes[0].find('t') >= 0:
            left_nodes, right_nodes = right_nodes, left_nodes
        #if len(left_nodes) != len(right_nodes):
            #pos = nx.bipartite_layout(graph_sub, left_nodes)
            #nx.draw(graph_sub, pos=pos, with_labels=True)
            #plt.savefig("test.png")
            
        bip_match = list(nx.bipartite.maximum_matching(graph_sub).items())
        if len_src >= len_tgt :
            for i in range(int(len(bip_match)/2)):
                t=0
                if bip_match[i][0].find('t') >= 0:
                    t = 1
                src_index = int(bip_match[i][t].replace("s",""))
                tgt_index = int(bip_match[i][t-1].replace("t",""))
                if not src_replace_table[src_index]:
                    src_idx_reordered[src_index] = "x"
                src_idx_reordered[tgt_index] = src_index
                src_replace_table[tgt_index] = True
                src_idx_remain.remove(src_index)
        else:
            for i in range(int(len(bip_match)/2)):
                t=0
                if bip_match[i][0].find('t') >= 0:
                    t = 1
                src_index = int(bip_match[i][t].replace("s",""))
                tgt_index = int(bip_match[i][t-1].replace("t",""))
                if not src_replace_table[src_index]:
                    src_idx_reordered[src_index] = "x"
                src_idx_reordered[tgt_index] = src_index
                src_replace_table[tgt_index] = True
                src_idx_remain.remove(src_index)

    if 'x' in src_idx_reordered :
        src_idx_reordered = list(filter(lambda x: x!='x', src_idx_reordered))
    for remain in src_idx_remain:
        if not remain in src_idx_reordered :
            if remain > 0 :
                src_idx_reordered.insert(src_idx_reordered.index(remain-1)+1,remain)
            else:
                src_idx_reordered.insert(0,remain)
    if len_src < len_tgt :
        src_idx_reordered = list(filter(lambda x: x<len_src, src_idx_reordered))
    if len(src_idx_reordered) != len_src:
        print("ERROR")
        import pdb;pdb.set_trace()

    for i in range(len_src):
        src_data_reordered[i] = src_line[src_idx_reordered[i]]
    return src_data_reordered


def build_graph(align_line):
    #add s-node and t-node
    src_node=[]
    tgt_node=[]
    src_tgt_edge=[]
    if len(align_line) <= 0:
        print("Error:align_line is empty")
    for i in range(len(align_line)):
        word_align = align_line[i].split("-")
        snode = "s" + str(word_align[0])
        tnode = "t" + str(word_align[1])
        src_node.append(snode)
        tgt_node.append(tnode)
        src_tgt_edge.append((snode,tnode))
    
    graph = nx.Graph()
    graph.add_nodes_from(src_node, bipartite=0)
    graph.add_nodes_from(tgt_node, bipartite=1)                 
    graph.add_edges_from(src_tgt_edge)
    return graph



def main():
    parser = argparse.ArgumentParser()
    #root_path = "examples/data/iwslt14.tokenized.de-en/"
    root_path = "examples/iwslt14.de-en.distilled.finetuned.train-3epoch/"
    output_name = "reorder"
    src_lang = "de"
    tgt_lang = "en"
    data_types = ["test"]
    output_data = []
    for data_type in data_types:
        align_lines, src_lines, tgt_lines = read_data(data_type,root_path, src_lang, tgt_lang)
        for line_id, align_line in tqdm(enumerate(align_lines), desc='processing', total=len(align_lines)):
            align_line = align_line.strip("\n").split(" ")
            src_line = src_lines[line_id].strip("\n").split(" ")
            tgt_line = tgt_lines[line_id].strip("\n").split(" ")
            len_src = len(src_line)
            len_tgt = len(tgt_line)

            #build graph
            graph = build_graph(align_line)
            #find connected component
            components = sorted(nx.connected_components(graph), key=len, reverse=True)
            src_reordered = reorder(graph, src_line, len_src, len_tgt, components)
            temp = (' '.join([str(elem) for elem in src_reordered])) + "\n"
            output_data.append(temp)
        
        #write to output path 
        output_path = root_path + data_type + "." + src_lang + "." + output_name
        with open(output_path, 'w', encoding="utf-8")as f:
            for data in output_data:
                f.write(data)




if __name__ == "__main__":
    main()
