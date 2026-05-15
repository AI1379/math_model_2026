import requests
import time
import json

def get_zhejiang_divisions():
    """获取浙江省所有市级、县级行政区划名称"""
    url = "https://dmfw.mca.gov.cn/9095/xzqh/getList"
    params = {"code": "330000000000", "maxLevel": 2}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    
    names = []
    for city in data.get("data", {}).get("children", []):
        city_name = city.get("name")
        if city_name:
            names.append(city_name)
        for district in city.get("children", []):
            district_name = district.get("name")
            if district_name:
                names.append(district_name)
    return names

def extract_coordinates(gdm):
    """从 GeoJSON 对象中提取代表性坐标点（第一个点）"""
    if not gdm:
        return None
    geom_type = gdm.get("type", "").lower()
    coords = gdm.get("coordinates")
    
    if geom_type == "point":
        return coords
    elif geom_type == "multipoint":
        return coords[0] if coords and len(coords) > 0 else None
    elif geom_type in ["linestring", "multilinestring"]:
        return coords[0] if coords else None
    elif geom_type == "polygon":
        return coords[0][0] if coords and coords[0] else None
    elif geom_type == "multipolygon":
        return coords[0][0][0] if coords and coords[0] and coords[0][0] else None
    else:
        print(f"未处理的几何类型: {geom_type}")
        return None

def search_place_coords(st_name, search_type="精确"):
    """查询单个地名的坐标，自动处理重名（优先返回浙江省内的结果）"""
    url = "https://dmfw.mca.gov.cn/9095/stname/listPub"
    params = {"stName": st_name, "searchType": search_type, "page": 1, "size": 5}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("records", [])
        
        # 1. 先从结果中筛选浙江省内的地名
        for record in records:
            if record.get("province_name") == "浙江省":
                standard_name = record.get("standard_name")
                gdm = record.get("gdm")
                coords = extract_coordinates(gdm)
                if coords:
                    return {"name": standard_name, "coordinates": coords}
        
        # 2. 如果没找到且原名称不含“浙江省”，则加上前缀重试
        if "浙江省" not in st_name:
            print(f"  未找到浙江省内的 {st_name}，尝试加上前缀...")
            return search_place_coords(f"浙江省{st_name}", search_type)
        
        # 3. 仍然找不到，返回 None
        return None
    except Exception as e:
        print(f"请求失败 {st_name}: {e}")
        return None

def batch_get_coords(names, delay=0.1):
    """批量查询地名坐标，返回列表，每个元素为 {"name": ..., "coordinates": ...}"""
    results = []
    total = len(names)
    for i, name in enumerate(names):
        print(f"进度：{i+1}/{total} - {name}")
        result = search_place_coords(name, "精确")
        if result is None:
            result = search_place_coords(name, "模糊")
        if result:
            results.append(result)
        time.sleep(delay)
    return results

if __name__ == "__main__":
    # 1. 获取所有市县名称
    all_names = get_zhejiang_divisions()
    print(f"共 {len(all_names)} 个地名待查询")
    
    # 2. 批量查询坐标（根据需要可限制数量进行测试）
    results = batch_get_coords(all_names)
    
    # 3. 输出为每行一个 JSON 对象，格式 {"name": "xxx", "gdm": [lng, lat]}
    with open("zhejiang_coords_per_line.json", "w", encoding="utf-8") as f:
        for item in results:
            # 将 coordinates 字段重命名为 gdm
            new_item = {"name": item["name"], "gdm": item["coordinates"]}
            f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
    
    print(f"已保存到 zhejiang_coords_per_line.json，共 {len(results)} 条记录")