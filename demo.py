import numpy as np
from src.dpwraistkinematics.wrist_tool import example

w = example()  # 示範幾何，需替換為實機尺寸

q = np.deg2rad([10, -5])
rho = w.ik(q)                  # 角度 → 致動器長度
q_back = w.fk(rho, seed=w.home) # 長度 → 角度

J = w.jacobian(q)
H = w.hessians(q)

qdot = np.array([0.1, 0.2])
qddot = np.array([0.02, -0.03])

rhodot = w.inverse_velocity(q, qdot)
rhoddot = w.inverse_acceleration(q, qdot, qddot)

force = w.torque_to_force(q, [1.0, 2.0])