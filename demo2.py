import json
import numpy as np
from dpjoint_wrist import Wrist

with open("geometry.json", encoding="utf-8") as f:
    w = Wrist(**json.load(f))

q = np.deg2rad([10.0, -5.0])
rho = w.ik(q)                      # 逆運動學：角度 -> 總長度
stroke = w.to_stroke(rho)          # 總長度 -> 行程
rho_again = w.from_stroke(stroke)  # 行程 -> 總長度
q_back = w.fk(rho, seed=w.home)    # 正運動學：總長度 -> 角度

print("總長度 m:", rho)
print("行程 m:", stroke)
print("正解 deg:", np.rad2deg(q_back))

q_previous = w.home.copy()

# 示範量測序列；實際應替換成感測器取得的兩支總長度。
rho_samples = [w.ik([0.1*t, -0.05*t]) for t in np.linspace(0, 1, 21)]
for rho_measured in rho_samples:
    q_current = w.fk(
        rho_measured,
        seed=q_previous,
        tol_m=1e-9,
        max_step_rad=np.deg2rad(2.0)
    )
    q_previous = q_current.copy()

q = np.deg2rad([10.0, -5.0])
w.ik(q)  # 先確認姿態對應的總長度在致動器限制內
qdot = np.array([0.1, 0.2])
qddot = np.array([0.02, -0.03])

# Inverse：手腕 -> 致動器
rhodot = w.inverse_velocity(q, qdot)
rhoddot = w.inverse_acceleration(q, qdot, qddot)

# Forward：致動器 -> 手腕
qdot_back = w.forward_velocity(q, rhodot)
qddot_back = w.forward_acceleration(q, rhodot, rhoddot)

# 固定座標系下的三維角速度、角加速度，兩者形狀都是 (3,)
omega, angular_acceleration = w.angular_state(q, qdot, qddot)

J = w.jacobian(q)                 # (2, 2)
H = w.hessians(q)                 # (2, 2, 2)
curvature = w.curvature(q, qdot)  # (2,)，即 Jdot @ qdot

force = np.array([20.0, 30.0])       # N，正值沿基座指向平台推
q_torque = w.force_to_torque(q, force)
force_back = w.torque_to_force(q, q_torque)

net_force, moment_about_O = w.actuator_wrench(q, force)