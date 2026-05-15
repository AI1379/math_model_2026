#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于铁路网（OSM 铁路线）计算浙江省64个市县之间的铁路旅行时间矩阵
所有铁路统一按 160 km/h 计算。
接驳段（城市到火车站）按汽车速度 30 km/h。
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import box
from math import radians, sin, cos, sqrt, asin

# ========== 配置 ==========
ox.settings.use_cache = True
ox.settings.log_console = False

# 速度设置 (米/小时)
SURFACE_SPEED = 30000          # 30 km/h，接驳段及无铁路时备用
RAILWAY_SPEED = 160000         # 160 km/h，火车速度统一设置

# ========== 加载坐标 ==========
def load_coordinates(json_path):
    coords = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            name = data['name']
            lon, lat = data['gdm']
            coords[name] = (lon, lat)
    return coords

# ========== 获取边界 ==========
def get_boundary():
    try:
        gdf = ox.geocode_to_gdf("Zhejiang Province, China")
        boundary = gdf.geometry.unary_union
    except Exception:
        west, south, east, north = 118.03, 27.14, 122.95, 31.18
        boundary = box(west, south, east, north)
    return boundary

def fetch_railway_graph_with_time():
    """下载铁路网，并为每条边添加 travel_time 权重（秒）"""
    boundary = get_boundary()
    railway_filter = '["railway"~"rail"]'
    print("正在下载铁路网 (可能较大，请耐心等待)...")
    G = ox.graph_from_polygon(
        boundary,
        custom_filter=railway_filter,
        network_type='all',
        simplify=True
    )
    print(f"原始铁路图：节点 {len(G.nodes)}，边 {len(G.edges)}")
    
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = ox.distance.great_circle_vec(
                G.nodes[u]['y'], G.nodes[u]['x'],
                G.nodes[v]['y'], G.nodes[v]['x']
            )
        travel_time_sec = data['length'] / RAILWAY_SPEED * 3600
        data['travel_time'] = travel_time_sec
    return G

# ========== 球面距离 ==========
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def find_nearest_node(G, lon, lat):
    nodes = list(G.nodes)
    def dist2(node):
        dx = G.nodes[node]['x'] - lon
        dy = G.nodes[node]['y'] - lat
        return dx*dx + dy*dy
    nearest = min(nodes, key=dist2)
    dist_m = haversine(lon, lat, G.nodes[nearest]['x'], G.nodes[nearest]['y'])
    return nearest, dist_m

# ========== 主流程 ==========
def main():
    # 加载坐标
    coords = load_coordinates("坐标.json")
    city_names = list(coords.keys())
    n = len(city_names)
    print(f"加载了 {n} 个市县坐标")
    
    # 获取铁路图
    G_rail = fetch_railway_graph_with_time()
    
    # 匹配最近铁路节点
    nearest_nodes = []
    dist_to_rail = []
    print("正在匹配各城市到最近铁路站点的距离...")
    for city in city_names:
        lon, lat = coords[city]
        node_id, dist_m = find_nearest_node(G_rail, lon, lat)
        nearest_nodes.append(node_id)
        dist_to_rail.append(dist_m)
        print(f"{city}: 最近铁路节点 {node_id}, 距离 {dist_m:.0f} 米")
    
    unique_nodes = list(set(nearest_nodes))
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    m = len(unique_nodes)
    print(f"共有 {m} 个唯一的铁路节点")
    
    # 计算铁路节点间最短旅行时间（秒）
    print("计算铁路节点间最短旅行时间...")
    rail_time_sec_matrix = np.full((m, m), np.inf)
    for i, src_node in enumerate(unique_nodes):
        lengths = nx.single_source_dijkstra_path_length(G_rail, src_node, weight='travel_time')
        for j, tgt_node in enumerate(unique_nodes):
            if tgt_node in lengths:
                rail_time_sec_matrix[i, j] = lengths[tgt_node]
    print("计算完成")
    
    # 构建时间矩阵（分钟）
    time_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                time_matrix[i, j] = 0
                continue
            idx_i = node_to_idx[nearest_nodes[i]]
            idx_j = node_to_idx[nearest_nodes[j]]
            rail_time_sec = rail_time_sec_matrix[idx_i, idx_j]
            if np.isfinite(rail_time_sec):
                access_sec_i = dist_to_rail[i] / SURFACE_SPEED * 3600
                egress_sec_j = dist_to_rail[j] / SURFACE_SPEED * 3600
                total_sec = access_sec_i + rail_time_sec + egress_sec_j
                time_min = total_sec / 60.0
            else:
                lon_i, lat_i = coords[city_names[i]]
                lon_j, lat_j = coords[city_names[j]]
                direct_m = haversine(lon_i, lat_i, lon_j, lat_j)
                time_min = (direct_m / SURFACE_SPEED) * 60
            time_matrix[i, j] = time_min
    
    df_time = pd.DataFrame(time_matrix, index=city_names, columns=city_names)
    df_time.to_csv("railway_travel_time_matrix_minutes.csv", float_format='%.1f')
    print("铁路旅行时间矩阵已保存为 railway_travel_time_matrix_minutes.csv")

if __name__ == "__main__":
    main()