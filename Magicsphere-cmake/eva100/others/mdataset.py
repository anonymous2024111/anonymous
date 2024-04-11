#!/usr/bin/env python3
import torch
import numpy as np
from scipy.sparse import *
import os

        
class MGCN_dataset_m16(torch.nn.Module):
    
    def __init__(self):
        super(MGCN_dataset_m16, self).__init__()
    
    def m_block_8_8_mr(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-8-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])

    def m_block_16_8_mr(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-16-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        
    def m_block_8_8_r(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-8-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])

    def m_block_16_8_r(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-16-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        
class MGCN_dataset_m32(torch.nn.Module):
    
    def __init__(self):
        super(MGCN_dataset_m32, self).__init__()
    
    def m_block_8_4_mr(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-tf32-8-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])

    def m_block_16_4_mr(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-tf32-16-1-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        
    def m_block_8_4_r(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-tf32-8-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])


    def m_block_16_4_r(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-tf32-16-1-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
 

        
        
class MGAT_dataset_m16(torch.nn.Module):
    
    def __init__(self):
        super(MGAT_dataset_m16, self).__init__()
    
    def m_block_8_16_mr(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-8-16-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])

    def m_block_8_16_r(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-8-16-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
        
    def m_block_16_8_mr(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-16-8-mr.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])

    def m_block_16_8_r(self, data, dimN): 
        self.graph = np.load('./dgl_dataset/block/' + data +'-fp16-16-8-r.npz')
        self.num_nodes_ori = self.graph['num_nodes_ori']-0
        self.num_nodes = self.graph['num_nodes']-0
        self.num_edges = self.graph['num_edges']-0
        
        self.row_pointers = torch.tensor(self.graph['row_pointers'])
        self.column_index = torch.tensor(self.graph['column_index'])
        self.degrees = torch.tensor(self.graph['degrees'])
