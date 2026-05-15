#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于高速公路网及直线距离计算浙江省64个市县之间的交通时间矩阵
高速段速度110 km/h，城区段速度40 km/h
输出CSV，时间保留一位小数
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import box
from math import radians, sin, cos, sqrt, asin

# ========== 1. 配置 ==========
ox.settings.use_cache = True
ox.settings.log_console = False

# 速度设置 (米/小时)
HIGHWAY_SPEED = 110000   # 110 km/h
SURFACE_SPEED = 40000    # 40 km/h

# ========== 2. 加载坐标 ==========
def load_coordinates(json_path):
    """加载坐标文件，每行一个JSON对象，返回字典 {name: (lon, lat)}"""
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

# ========== 3. 获取高速公路图 ==========
def get_boundary():
    """获取浙江省行政边界，失败时使用矩形框"""
    try:
        gdf = ox.geocode_to_gdf("Zhejiang Province, China")
        boundary = gdf.geometry.unary_union
    except Exception:
        west, south, east, north = 118.03, 27.14, 122.95, 31.18
        boundary = box(west, south, east, north)
    return boundary

def fetch_highway_graph():
    """下载并返回高速公路图（含高速及快速路）"""
    boundary = get_boundary()
    highway_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link"]'
    G = ox.graph_from_polygon(
        boundary,
        custom_filter=highway_filter,
        network_type='drive',
        simplify=True
    )
    # 确保边有长度信息
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = ox.distance.great_circle_vec(
                G.nodes[u]['y'], G.nodes[u]['x'],
                G.nodes[v]['y'], G.nodes[v]['x']
            )
    return G

# ========== 4. 球面距离计算（米）==========
def haversine(lon1, lat1, lon2, lat2):
    """返回两点之间球面距离（米）"""
    R = 6371000
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def find_nearest_node(G, lon, lat):
    """在图G中找到距离(lon,lat)最近的节点，返回(node_id, 距离_米)"""
    nodes = list(G.nodes)
    def dist2(node):
        dx = G.nodes[node]['x'] - lon
        dy = G.nodes[node]['y'] - lat
        return dx*dx + dy*dy
    nearest = min(nodes, key=dist2)
    dist_m = haversine(lon, lat, G.nodes[nearest]['x'], G.nodes[nearest]['y'])
    return nearest, dist_m

# ========== 5. 主流程 ==========
def main():
    # 读取坐标
    coords = load_coordinates("坐标.json")
    city_names = list(coords.keys())
    n = len(city_names)
    print(f"加载了 {n} 个市县坐标")

    # 获取高速图
    print("正在下载高速公路网...")
    G_highway = fetch_highway_graph()
    print(f"高速公路图: {G_highway.number_of_nodes()} 节点, {G_highway.number_of_edges()} 边")

    # 为每个城市找到最近高速节点和直线距离
    nearest_nodes = []
    dist_to_highway = []  # 单位：米
    for city in city_names:
        lon, lat = coords[city]
        node_id, dist_m = find_nearest_node(G_highway, lon, lat)
        nearest_nodes.append(node_id)
        dist_to_highway.append(dist_m)
        print(f"{city}: 最近高速节点 {node_id}, 距离 {dist_m:.0f} 米")

    # 获取所有唯一的高速节点ID
    unique_nodes = list(set(nearest_nodes))
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    m = len(unique_nodes)
    print(f"共有 {m} 个唯一的高速节点")

    # 计算这些唯一节点之间的最短路径长度（米）
    print("计算高速节点间最短路径...")
    highway_dist_matrix = np.full((m, m), np.inf)
    for i, src_node in enumerate(unique_nodes):
        lengths = nx.single_source_dijkstra_path_length(G_highway, src_node, weight='length')
        for j, tgt_node in enumerate(unique_nodes):
            if tgt_node in lengths:
                highway_dist_matrix[i, j] = lengths[tgt_node]
    print("高速节点间最短路径计算完成")

    # 构建最终的时间矩阵（分钟）
    time_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                time_matrix[i, j] = 0
                continue
            idx_i = node_to_idx[nearest_nodes[i]]
            idx_j = node_to_idx[nearest_nodes[j]]
            highway_path_m = highway_dist_matrix[idx_i, idx_j]

            if np.isfinite(highway_path_m):
                time_min = (dist_to_highway[i] / SURFACE_SPEED +
                            highway_path_m / HIGHWAY_SPEED +
                            dist_to_highway[j] / SURFACE_SPEED) * 60
            else:
                lon_i, lat_i = coords[city_names[i]]
                lon_j, lat_j = coords[city_names[j]]
                direct_m = haversine(lon_i, lat_i, lon_j, lat_j)
                time_min = (direct_m / SURFACE_SPEED) * 60
            time_matrix[i, j] = time_min

    # 转换为DataFrame
    df_time = pd.DataFrame(time_matrix, index=city_names, columns=city_names)
    # 保存CSV，保留一位小数
    df_time.to_csv("travel_time_matrix_minutes.csv", float_format='%.1f')
    print("时间矩阵已保存为 travel_time_matrix_minutes.csv (时间保留一位小数)")

if __name__ == "__main__":
    main()