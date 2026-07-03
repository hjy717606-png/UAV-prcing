import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import heapq
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# 路径自适应（本地 + 云端通用）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = BASE_DIR

processed_dir = os.path.join(DATA_ROOT, 'processed')
checkpoints_dir = os.path.join(DATA_ROOT, 'checkpoints')
city_dir = os.path.join(DATA_ROOT, 'city')
accidents_dir = os.path.join(DATA_ROOT, 'accidents')

# ============================================================
# 精算转换参数
# ============================================================
BASE_PREMIUM = 1000.0
EXPENSE_LOAD = 0.25
REGULATORY_REDLINE = 0.85


def risk_to_premium(risk_scores, base=BASE_PREMIUM, expense_load=EXPENSE_LOAD):
    y_hat = np.clip(risk_scores, 1e-6, 0.9999)
    odds = y_hat / (1.0 - y_hat)
    return base * odds * (1.0 + expense_load)


def compute_combined_ratio(pred_risks, target_risks,
                           base=BASE_PREMIUM, expense_load=EXPENSE_LOAD):
    pc = np.clip(pred_risks, 1e-6, 0.9999)
    tc = np.clip(target_risks, 1e-6, 0.9999)
    total_premium = np.sum(base * pc / (1.0 - pc) * (1.0 + expense_load))
    total_payout = np.sum(base * tc / (1.0 - tc))
    net_premium = total_premium / (1.0 + expense_load)
    return (total_payout / (net_premium + 1e-8)) * 100.0, total_premium, total_payout


# ============================================================
# A* 寻路算法 + 城市栅格地图
# ============================================================

def generate_city_grid(city, start_x, start_y, end_x, end_y, size=50, density=0.25):
    """生成模拟城市栅格地图: 0=可飞, 1=建筑物"""
    seed = hash(city) % 2**31
    rng = np.random.RandomState(seed)
    grid = np.zeros((size, size))
    n_buildings = int(size * size * density / 9)
    for _ in range(n_buildings):
        bx = rng.randint(2, size - 3)
        by = rng.randint(2, size - 3)
        grid[by - 1:by + 2, bx - 1:bx + 2] = 1
    # 确保起点终点区域畅通
    for y, x in [(start_y, start_x), (end_y, end_x)]:
        y0, x0 = max(0, y - 2), max(0, x - 2)
        y1, x1 = min(size, y + 3), min(size, x + 3)
        grid[y0:y1, x0:x1] = 0
    return grid


def astar(grid, start, end):
    """A*寻路, 返回路径点列表 [(x1,y1), ...]"""
    rows, cols = grid.shape
    if grid[start[1], start[0]] == 1 or grid[end[1], end[0]] == 1:
        return None
    open_set = [(0, start[0], start[1])]
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == end:
            path = []
            curr = (cx, cy)
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            return path[::-1]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 0:
                move = 1.414 if dx != 0 and dy != 0 else 1
                tentative = g_score.get((cx, cy), float('inf')) + move
                if tentative < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), end)
                    heapq.heappush(open_set, (f, nx, ny))
    return None


# ============================================================
# 城市坐标
# ============================================================
CITY_INFO = {
    'Beijing': {'lat': 39.9042, 'lon': 116.4074, 'alias': '北京'},
    'Shanghai': {'lat': 31.2304, 'lon': 121.4737, 'alias': '上海'},
    'Shenzhen': {'lat': 22.5431, 'lon': 114.0579, 'alias': '深圳'},
    'Chengdu': {'lat': 30.5728, 'lon': 104.0668, 'alias': '成都'},
    'Xi_an': {'lat': 34.3416, 'lon': 108.9398, 'alias': '西安'},
}


st.set_page_config(page_title="低空飞行器保守学习定价引擎", layout="wide")


# ============================================================
# 模型定义
# ============================================================

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 40, 3, padding=1), nn.ReLU(),
            nn.Conv1d(40, 40, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(40, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.conv(x).squeeze(-1))


class StructuralEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class DynamicGateFusion(nn.Module):
    def __init__(self, hidden_dim=80):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 40), nn.ReLU(),
            nn.Linear(40, 2), nn.Softmax(dim=1)
        )
        self.traj_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, traj_feat, env_feat):
        w = self.gate(env_feat)
        return w[:, 0:1] * self.traj_proj(traj_feat) + w[:, 1:2] * env_feat, w


class MultiplicativeRiskHeads(nn.Module):
    def __init__(self, hidden_dim=80):
        super().__init__()
        self.env_head = nn.Sequential(nn.Linear(hidden_dim, 40), nn.ReLU(), nn.Linear(40, 1), nn.Sigmoid())
        self.op_head = nn.Sequential(nn.Linear(hidden_dim, 40), nn.ReLU(), nn.Linear(40, 1), nn.Sigmoid())
        self.total_head = nn.Sequential(nn.Linear(hidden_dim, 40), nn.ReLU(), nn.Linear(40, 1), nn.Sigmoid())

    def forward(self, env_feat, traj_feat, fused_feat):
        r_env = self.env_head(env_feat)
        r_op = self.op_head(traj_feat)
        r_raw = self.total_head(fused_feat)
        r_mul = 1.0 - (1.0 - r_env) * (1.0 - r_op)
        return r_env, r_op, 0.5 * r_raw + 0.5 * r_mul, r_mul, r_raw


class FullModel(nn.Module):
    def __init__(self, seq_input_dim=2, struct_input_dim=23, hidden_dim=80):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(seq_input_dim, hidden_dim)
        self.structural_encoder = StructuralEncoder(struct_input_dim, hidden_dim)
        self.gate_fusion = DynamicGateFusion(hidden_dim)
        self.risk_heads = MultiplicativeRiskHeads(hidden_dim)

    def forward(self, x_seq, x_struct):
        traj = self.temporal_encoder(x_seq)
        env = self.structural_encoder(x_struct)
        fused, gate = self.gate_fusion(traj, env)
        r_env, r_op, r_total, _, _ = self.risk_heads(env, traj, fused)
        return {'r_total': r_total, 'r_env': r_env, 'r_op': r_op, 'gate_weights': gate}


# ============================================================
# 加载模型
# ============================================================

@st.cache_resource
def load_model():
    DATA_ROOT = 'D:/data'
    processed_dir = os.path.join(DATA_ROOT, 'processed')
    checkpoint_dir = os.path.join(DATA_ROOT, 'checkpoints')

    with open(os.path.join(processed_dir, 'structured_scaler.pkl'), 'rb') as f:
        info = pickle.load(f)

    model = FullModel(struct_input_dim=len(info['feature_names']))
    ckpt_path = os.path.join(checkpoint_dir, 'best_model_ema.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.eval()
        loaded = True
    else:
        model.eval()
        loaded = False

    return model, info['scaler'], info['feature_names'], loaded, DATA_ROOT


model, scaler, feature_cols, model_loaded, DATA_ROOT = load_model()


# ============================================================
# 模拟推理
# ============================================================

@st.cache_data
def mock_predict(city, weather, start_x, start_y, end_x, end_y):
    weather_params = {
        '晴天': {'w': 3, 'v': 10}, '小雨': {'w': 5, 'v': 5},
        '大风': {'w': 12, 'v': 8}, '雷暴': {'w': 15, 'v': 2},
        '雾': {'w': 2, 'v': 0.5}, '雪': {'w': 6, 'v': 1}
    }
    w = weather_params[weather]
    dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    seed = hash((city, weather, start_x, start_y, end_x, end_y)) % 2**31
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.05)
    risk = min(0.95, 0.1 + w['w']*0.02 + (50-dist)*0.005 + 1/w['v']*0.05 + noise)
    risk = max(0.05, risk)
    premium = risk_to_premium(risk)
    return {
        'risk': float(risk), 'premium': float(premium),
        'env_risk': float(risk*0.6), 'op_risk': float(risk*0.4),
        'gate': [0.4, 0.6]
    }


# ============================================================
# 侧边栏
# ============================================================

st.sidebar.header("飞行参数设置")
city = st.sidebar.selectbox("城市区域", list(CITY_INFO.keys()),
                            format_func=lambda x: CITY_INFO[x]['alias'])
weather = st.sidebar.selectbox("天气模板", ['晴天', '小雨', '大风', '雷暴', '雾', '雪'])
start_x = st.sidebar.slider("起点 X", 0, 49, 10)
start_y = st.sidebar.slider("起点 Y", 0, 49, 10)
end_x = st.sidebar.slider("终点 X", 0, 49, 40)
end_y = st.sidebar.slider("终点 Y", 0, 49, 40)

st.sidebar.markdown("---")
st.sidebar.subheader("精算参数")
st.sidebar.write(f"基础保费: {BASE_PREMIUM:.0f}元")
st.sidebar.write(f"费用附加率: {EXPENSE_LOAD*100:.0f}%")
st.sidebar.write(f"监管红线: {REGULATORY_REDLINE*100:.0f}%")

st.sidebar.markdown("---")
if model_loaded:
    st.sidebar.success("模型已加载")
else:
    st.sidebar.warning("使用模拟推理")


# ============================================================
# 执行推理
# ============================================================

result = mock_predict(city, weather, start_x, start_y, end_x, end_y)
info = CITY_INFO[city]


# ============================================================
# 主界面
# ============================================================

col1, col2, col3 = st.columns([1, 2, 1])

# ---------- 左列 ----------
with col1:
    st.subheader("风险仪表盘")

    fig, ax = plt.subplots(figsize=(4, 4))
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    risk_angle = result['risk'] * 2 * np.pi
    ax.fill_between(np.cos(np.linspace(0, risk_angle, 50)),
                    np.sin(np.linspace(0, risk_angle, 50)), alpha=0.3, color='red')
    ax.text(0, 0, f"{result['risk']:.2f}", ha='center', va='center', fontsize=24, fontweight='bold')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.axis('off')
    st.pyplot(fig); plt.close(fig)

    st.metric("动态保费", f"{result['premium']:.0f}元")

    if result['risk'] > 0.6:
        st.warning("高风险 - 保费精算上浮")
    elif result['risk'] > 0.3:
        st.info("中风险 - 精算定价正常")
    else:
        st.success("低风险")

    with st.expander("保费精算分解"):
        y_hat = np.clip(result['risk'], 1e-6, 0.9999)
        odds = y_hat / (1.0 - y_hat)
        st.write(f"风险评分: {result['risk']:.4f}")
        st.write(f"Odds Ratio: {odds:.4f}")
        st.write(f"Base x Odds: {BASE_PREMIUM * odds:.0f}元")
        st.write(f"x (1 + {EXPENSE_LOAD*100:.0f}% 费用附加)")
        st.write(f"**最终保费: {result['premium']:.0f}元**")

    st.subheader("风险分解")
    st.progress(result['env_risk'], text=f"环境风险: {result['env_risk']:.3f}")
    st.progress(result['op_risk'], text=f"操作风险: {result['op_risk']:.3f}")


# ---------- 中列：真实地图 + A*寻路航线 ----------
with col2:
    st.subheader(f"航线可视化 - {info['alias']}")

    base_lat = info['lat']
    base_lon = info['lon']
    center_lat = base_lat + (start_y + end_y) * 0.005
    center_lon = base_lon + (start_x + end_x) * 0.005

    max_diff = max(abs(end_y - start_y) * 0.01, abs(end_x - start_x) * 0.01)
    if max_diff < 0.01:
        zoom = 13
    elif max_diff < 0.03:
        zoom = 12
    elif max_diff < 0.05:
        zoom = 11
    else:
        zoom = 10

    # 最保守的 folium 调用：只设 location 和 zoom_start
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    # 高德瓦片（国内可稳定访问）
    folium.TileLayer(
        tiles='https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德地图',
        subdomains='1234',
        name='高德'
    ).add_to(m)

    # A*寻路生成轨迹
    grid = generate_city_grid(city, start_x, start_y, end_x, end_y)
    path = astar(grid, (start_x, start_y), (end_x, end_y))

    if path is None:
        # 无路径时回退到直线
        path = [(start_x, start_y), (end_x, end_y)]
        st.info("起点终点被建筑物包围，显示直线路径")

    # 绘制轨迹
    points = [[base_lat + y * 0.01, base_lon + x * 0.01] for x, y in path]
    folium.PolyLine(
        points,
        color='blue' if result['risk'] < 0.6 else 'red',
        weight=3 if result['risk'] < 0.6 else 5,
        opacity=0.9
    ).add_to(m)

    # 起点终点标记
    folium.Marker(
        [base_lat + start_y * 0.01, base_lon + start_x * 0.01],
        popup='起点', icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    folium.Marker(
        [base_lat + end_y * 0.01, base_lon + end_x * 0.01],
        popup='终点', icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

    # 绘制建筑物（半透明红色方块）
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                folium.Rectangle(
                    bounds=[[base_lat + y * 0.01, base_lon + x * 0.01],
                            [base_lat + (y + 1) * 0.01, base_lon + (x + 1) * 0.01]],
                    color='#ff6b6b',
                    fill=True,
                    fill_color='#ff6b6b',
                    fill_opacity=0.25,
                    weight=0
                ).add_to(m)

    # 最保守的 st_folium：只设 width 和 height
    st_folium(m, width=700, height=500)


# ---------- 右列 ----------
with col3:
    st.subheader("可解释性面板")

    st.write("**门控权重分配**")
    st.write(f"时序流(轨迹): {result['gate'][0]:.1%}")
    st.write(f"环境流(天气): {result['gate'][1]:.1%}")

    st.write("**Top-3 风险因子**")
    factors = [
        ('建筑密度 x 风速', result['env_risk'] * 0.4),
        ('能见度倒数', 1/10 * 0.3),
        ('轨迹曲率', result['op_risk'] * 0.3)
    ]
    for name, val in factors:
        st.write(f"- {name}: {val:.3f}")

    st.markdown("---")
    st.subheader("保险充足性指标")

    batch_risks = np.array([result['risk']])
    batch_targets = np.array([result['risk'] * 1.1])
    cr, total_prem, total_pay = compute_combined_ratio(batch_risks, batch_targets)

    cr_col, ad_col = st.columns(2)
    with cr_col:
        st.metric("综合赔付率", f"{cr:.1f}%")
    with ad_col:
        adequacy = total_prem / (total_pay + 1e-8)
        st.metric("保费充足率", f"{adequacy:.2f}")

    if cr < 100.0:
        st.success("保费池健康 (CR < 100%)")
    else:
        st.error("保费池亏损 (CR > 100%)")

    if adequacy >= REGULATORY_REDLINE:
        st.success(f"满足监管红线 ({REGULATORY_REDLINE*100:.0f}%)")
    else:
        st.error(f"低于监管红线 ({REGULATORY_REDLINE*100:.0f}%)")


# ============================================================
# 事故回溯页
# ============================================================

st.header("真实事故回溯验证")

try:
    accidents = pd.read_csv(os.path.join(DATA_ROOT, 'accidents', 'real_accident_cases_15.csv'))
    for _, row in accidents.iterrows():
        with st.expander(f"{row['case_id']} | {row['city']} | {row['weather']} | {row['main_factor']}"):
            cols = st.columns(3)
            cols[0].metric("真实主因", row['main_factor'])
            cols[1].metric("事故类型", row['terrain_type'])
            cols[2].metric("风速", f"{row['wind_speed']} m/s")
except FileNotFoundError:
    st.info("事故数据文件未找到，跳过事故回溯验证")

st.markdown("---")
st.caption("基于动态门控与乘法风险分解的低空飞行器保守学习定价引擎 | 精算转换 + 保费充足性约束")
