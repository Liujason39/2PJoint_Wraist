import numpy as np
from src.dpjoint_wrist.wrist_tool import example

np.set_printoptions(precision=6, suppress=True)


def print_result(name, value, unit=""):
    """顯示數值或矩陣，並標示單位。"""
    label = f"{name} [{unit}]" if unit else name
    print(f"\n{label}:")
    print(np.asarray(value))


w = example()  # 確認 example() 已替換為實機幾何參數

q = np.deg2rad([0, 0])
# initial length = 0.23302m
# 位置
rho = w.ik(q)
q_back = w.fk(rho, seed=w.home)

# 導數矩陣
J = w.jacobian(q)
H = w.hessians(q)

# 手腕角速度、角加速度
qdot = np.array([0.1, 0.2])       # rad/s
qddot = np.array([0.02, -0.03])   # rad/s²

# 致動器線速度、線加速度
rhodot = w.inverse_velocity(q, qdot)             # m/s
rhoddot = w.inverse_acceleration(q, qdot, qddot)  # m/s²

# 手腕廣義力矩 → 致動器軸向力
torque = np.array([1.0, 2.0])  # N·m
force = w.torque_to_force(q, torque)

# 輸出
print("角度順序：[alpha, beta]")
print("致動器順序：[1, 2]")

print_result("輸入手腕角度 q", q, "rad")
print_result("輸入手腕角度 q", np.rad2deg(q), "deg")
print_result("致動器總長度 rho", rho, "m")
print_result("致動器總長度 rho", rho * 1000, "mm")
print_result("正運動學角度 q_back", np.rad2deg(q_back), "deg")
print_result("正逆解角度誤差 q_back - q", q_back - q, "rad")

print_result("Jacobian J（列：致動器；欄：alpha、beta）", J, "m/rad")
for i, Hi in enumerate(H, start=1):
    print_result(
        f"致動器 {i} 的 Hessian（列、欄皆為 alpha、beta）",
        Hi,
        "m/rad²",
    )

print_result("手腕角速度 qdot", qdot, "rad/s")
print_result("手腕角加速度 qddot", qddot, "rad/s²")
print_result("致動器線速度 rhodot", rhodot, "m/s")
print_result("致動器線加速度 rhoddot", rhoddot, "m/s²")
print_result("手腕廣義力矩 torque", torque, "N·m")
print_result("致動器軸向力 force", force, "N")