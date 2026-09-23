# 兩自由度並聯手腕 Python 工具

`wrist_tool.py` 用於中央十字軸承加兩支直線致動器（P joint）的手腕機構，提供正逆運動學、解析 Jacobian 與 Hessian、速度及加速度轉換、力與力矩映射，以及由使用者提供參數的動力學計算介面。

> 內建幾何尺寸僅供示範，並非實機量測值。本工具不包含硬體通訊或馬達驅動。

## 1. 安裝與快速執行

需要 Python 3、NumPy、SciPy。將本 README 與 `wrist_tool.py` 放在同一資料夾。

```bash
python -m pip install numpy scipy
python wrist_tool.py
```

執行內建數值驗證：

```bash
python wrist_tool.py --test
```

讀取自訂幾何設定：

```bash
python wrist_tool.py --config geometry.json
```

`--config` 會在設定的 `q_home` 執行示範，輸出長度、Jacobian、Hessian、速度、加速度及力等資料。它不是任意軌跡的命令列求解器；一般計算請使用下方 Python API。

## 2. 模型、座標與單位

### 機構假設

- 中央十字軸承固定手腕旋轉中心，平台具有兩個旋轉自由度。
- 兩支致動器的基座中心固定，致動器方向可隨平台運動改變。
- 兩端接頭允許所需的空間擺動，支鏈以兩端距離描述。
- 若實際基座或平台接頭只有單軸鉸鏈，可能存在額外方向／平面約束，不能直接視為本模型。
- 未建模碰撞、接頭擺角、彈性、間隙及硬體速度／推力限制。

### 固定座標系

| 名稱 | 定義 |
| --- | --- |
| 原點 O | 中央十字軸承的交點 |
| +Z | 沿前臂朝手掌 |
| +X | 手腕橫向，朝其中一側 |
| +Y | 依右手定則，eY = eZ × eX |

平台自身座標系在幾何零姿態與固定座標系重合，之後跟隨平台旋轉。

角度向量為 `q = [alpha, beta]`，旋轉矩陣固定採用：

$$
R(q)=R_y(\beta)R_x(\alpha).
$$

這對應外層固定 Y 軸轉動 beta，內層隨動 X′ 軸轉動 alpha。角度正向遵循右手定則。若實機軸序相反，不能只交換輸入名稱，必須修改旋轉矩陣、導數及角速度映射。

| 量 | 單位 |
| --- | --- |
| 座標、總長度、伸縮行程 | m |
| 角度、角速度、角加速度 | rad、rad/s、rad/s² |
| 伸縮速度、加速度 | m/s、m/s² |
| 致動器軸向力 | N |
| 廣義力矩、三維力矩 | N·m |
| 等效慣性矩陣 M | kg·m² |

`q_home` 是已知初始裝配姿態／求解初值，不會重新定義幾何零位，也不會自動加到輸入角度上。

## 3. JSON 幾何設定

建立 `geometry.json`，貼入以下合法 JSON：

```json
{
  "a": [
    [-.033,.00,-.2315],
    [.033,.00,-.2315]
  ],
  "b": [
    [-.0695/2,-0.02,0],
    [.0695/2,-0.02,0]
  ],
  "q_min": [-0.6108652382, -0.6108652382],
  "q_max": [0.6108652382, 0.6108652382],
  "q_home": [0.0, 0.0],
  "rho_min": [.18302,.18302],
  "rho_max": [.28302,.28302],
  "rho_offset": [0.23237,0.23237]
}
```

| 欄位 | 形狀 | 定義 | 是否必填 |
| --- | --- | --- | --- |
| `a` | (2, 3) | 兩個基座中心在固定座標系中的 xyz | 是 |
| `b` | (2, 3) | 兩個平台接點在平台自身座標系中的 xyz | 是 |
| `q_min` | (2,) | `[alpha_min, beta_min]` | 是 |
| `q_max` | (2,) | `[alpha_max, beta_max]` | 是 |
| `q_home` | (2,) | 初始裝配角度 `[alpha, beta]` | 是 |
| `rho_min` | (2,) | 兩支致動器的最小總長度；預設 `[0, 0]`，實際長度仍須大於零 | 否 |
| `rho_max` | (2,) | 最大總長度；省略表示沒有設定上限 | 否 |
| `rho_offset` | (2,) | 行程讀值為零時的總長度；預設 `[0, 0]` | 否 |

注意：

- `a[0]` 與 `b[0]` 對應第一支致動器，第二列同理。
- `a` 是固定座標；`b` 是平台局部座標，不是旋轉後的座標。
- 總長度 rho 是兩端接頭中心的距離；行程 `stroke = rho - rho_offset`。
- `rho_offset` 不一定等於最小總長度；示範以零姿態總長度當作行程零點，因此行程可為負。
- 角度限制示範為 ±35°，已換算為 rad。JSON 不能寫註解或 `np.deg2rad(...)`。
- 必須逐軸滿足 `q_min < q_max`；`q_home` 必須符合角度與總長度限制。
- Python API 的形狀 `(2,)` 表示 `[x, y]`，不是 `[[x], [y]]`。

## 4. 載入與正逆運動學

```python
import json
import numpy as np
from wrist_tool import Wrist

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
```

未準備 JSON 時可使用 `from wrist_tool import example; w = example()`。

### 連續正解與分支追蹤

```python
import numpy as np
from wrist_tool import example

w = example()
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
```

`fk` 使用有界非線性最小平方法求解 `h(q) = rho`，並檢查最大絕對長度殘差。

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `seed` | `w.home` | 初始猜測；追蹤時傳入前次姿態 |
| `tol_m` | `1e-9` | 可接受的最大單支長度殘差，依量測與模型精度設定 |
| `max_step_rad` | `None` | 相對 seed 的最大單軸角度變化；省略則不檢查 |

`max_step_rad` 是求解後的拒絕條件，不是求解器的局部搜尋邊界。角度限制和初值可協助選擇裝配分支，但不保證全域唯一；跨越奇異處或大幅跳動時仍可能無法追蹤。FK 在求得奇異／過度病態的姿態時也會拒絕輸出，即使長度殘差很小。

## 5. 速度、加速度與三維角速度

以下使用前節已建立的 `w`：

```python
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
```

`forward_acceleration` 的第二個參數是致動器速度 `rhodot`，不是 `qdot`。常數長度偏移不影響導數，所以行程速度／加速度與總長度速度／加速度相同。

## 6. 力、力矩與動力學

### 靜態／瞬時傳動映射

```python
force = np.array([20.0, 30.0])       # N，正值沿基座指向平台推
q_torque = w.force_to_torque(q, force)
force_back = w.torque_to_force(q, q_torque)

net_force, moment_about_O = w.actuator_wrench(q, force)
```

`q_torque` 是與 `[alpha, beta]` 共軛的兩個廣義力矩，不是固定座標系的 `[Mx, My]`。`actuator_wrench` 回傳兩個三維向量，表示致動器對平台施加的合力與繞 O 的力矩，不包含十字軸承反力及反力矩。

### 動力學介面

```python
# 僅示範介面；以下 M 與 bias 並非此機構的辨識結果。
M = np.array([[0.10, 0.01], [0.01, 0.20]])
bias = np.array([0.2, 0.3])       # 科氏 + 離心 + 重力 + 摩擦，N*m
external = np.array([0.1, 0.0])   # 外部廣義力矩，N*m

force_required = w.inverse_dynamics(q, qddot, M, bias, external)
qddot_result = w.forward_dynamics(q, force_required, M, bias, external)
```

使用者必須在當前 `q`、`qdot` 下計算 `M` 和 `bias`，並包含模型決定納入的負載與移動部件。工具不會從幾何自動推算它們，因此動力學函式沒有單獨的 `qdot` 參數。`M` 必須為對稱正定矩陣。

輸出的 `force_required` 是直線致動器軸向力，不是螺桿馬達扭矩；螺桿導程、減速比、效率及馬達慣量未包含在工具中。

## 7. 計算公式

### 幾何與位置

$$
p_i=Rb_i,\quad d_i=p_i-a_i,\quad \rho_i=h_i(q)=\|d_i\|,\quad u_i=d_i/\rho_i.
$$

逆運動學直接計算兩端距離；正運動學在角度限制內求解：

$$
\min_q\frac12\|h(q)-\rho_{\rm measured}\|^2.
$$

### Jacobian 與 Hessian

令 $p_{i,j}=\partial p_i/\partial q_j$，$p_{i,jk}=\partial^2p_i/(\partial q_j\partial q_k)$，則：

$$
J_{ij}=u_i^T p_{i,j}.
$$

$$
\frac{\partial u_i}{\partial q_k}
=\frac{1}{\rho_i}(I-u_i u_i^T)p_{i,k}.
$$

$$
(H_i)_{jk}
=\frac{p_{i,j}^T(I-u_i u_i^T)p_{i,k}}{\rho_i}
+u_i^T p_{i,jk}.
$$

對非零桿長的平滑函數，混合偏導相等，因此 H 的後兩個維度對稱。這不代表旋轉矩陣的乘法順序可以交換。

### 速度與加速度

$$
\dot\rho=J\dot q,\qquad
\ddot\rho=J\ddot q+\dot J\dot q.
$$

$$
\dot J\dot q=
\begin{bmatrix}\dot q^T H_1\dot q\\\dot q^T H_2\dot q\end{bmatrix}.
$$

非奇異時：

$$
\dot q=J^{-1}\dot\rho,\qquad
\ddot q=J^{-1}(\ddot\rho-\dot J\dot q).
$$

### 固定座標系下的角速度

$$
\omega=E\dot q,\qquad
E=\begin{bmatrix}\cos\beta&0\\0&1\\-\sin\beta&0\end{bmatrix}.
$$

E 不含 alpha，因為繞內層軸旋轉不會改變該軸本身的方向；該方向由外層 beta 決定。這不表示姿態或長度不受 alpha 影響。

$$
\dot\omega=E\ddot q+
\begin{bmatrix}
-\sin\beta\,\dot\alpha\dot\beta\\
0\\
-\cos\beta\,\dot\alpha\dot\beta
\end{bmatrix}.
$$

### 力與動力學

由功率守恆 $f^T\dot\rho=\tau_q^T\dot q$：

$$
\tau_q=J^Tf,\qquad f=J^{-T}\tau_q.
$$

動力學採用：

$$
M\ddot q+\mathrm{bias}=J^Tf+\tau_{\rm ext}.
$$

因此：

$$
f=J^{-T}(M\ddot q+\mathrm{bias}-\tau_{\rm ext}),
$$

$$
\ddot q=M^{-1}(J^Tf+\tau_{\rm ext}-\mathrm{bias}).
$$

程式使用線性方程求解，不直接計算矩陣反矩陣。

## 8. API 速查

下表中 q、rho、速度、加速度、force、torque 均為兩元素向量，除非另有註明。

| 函式 | 回傳／用途 |
| --- | --- |
| `Wrist(a, b, q_min, q_max, q_home, rho_min=None, rho_max=None, rho_offset=None)` | 建立模型 |
| `example()` | 建立內建示範模型 |
| `ik(q)` | 總長度 rho |
| `fk(rho, seed=None, tol_m=1e-9, max_step_rad=None)` | 手腕角度 q |
| `to_stroke(rho)` / `from_stroke(stroke)` | 行程與總長度轉換 |
| `geometry(q)` | `(rho, J, H, p, u)`；p、u 的形狀皆為 (2,3) |
| `jacobian(q)` / `hessians(q)` | 解析 J / H |
| `curvature(q, qdot)` | Jdot @ qdot |
| `inverse_velocity(q, qdot)` | rhodot |
| `forward_velocity(q, rhodot)` | qdot |
| `inverse_acceleration(q, qdot, qddot)` | rhoddot |
| `forward_acceleration(q, rhodot, rhoddot)` | qddot |
| `angular_state(q, qdot, qddot)` | `(omega, angular_acceleration)`，各為 (3,) |
| `force_to_torque(q, force)` | 廣義力矩 |
| `torque_to_force(q, torque)` | 致動器軸向力 |
| `actuator_wrench(q, force)` | `(net_force, moment_about_O)`，各為 (3,) |
| `inverse_dynamics(q, qddot, M, bias, external=(0,0))` | 軸向力 |
| `forward_dynamics(q, force, M, bias, external=(0,0))` | qddot |
| `diagnostics(q)` | `singular_values`、`condition` |

模組層級的 `rotations(q)` 回傳 R、一階導數、二階導數，形狀依序為 (3,3)、(2,3,3)、(2,2,3,3)。

## 9. 錯誤處理、奇異性與限制

```python
try:
    q_estimated = w.fk(rho, seed=w.home)
except ValueError as error:
    print("求解未通過：", error)
```

| 錯誤訊息關鍵字 | 意義與檢查方向 |
| --- | --- |
| `expected finite shape` | 輸入形狀錯誤，或含 NaN／無限值 |
| `angle outside limits` | 檢查 rad／deg、角度正負與設定上下限 |
| `length outside limits` | 檢查 m／mm，以及是否把行程誤當總長度 |
| `zero-length actuator` | 接點重合，方向向量無法定義 |
| `FK failed/unreachable or incompatible data` | 可能不可達、初值不合適、量測／模型不一致，或求解器未收斂 |
| `FK step too large` | 求解結果相對 seed 超過 max_step_rad |
| `singular/ill-conditioned Jacobian` | 當前姿態不能可靠執行精確逆映射 |
| `M must be symmetric positive definite` | 動力學慣性矩陣不符合要求 |

目前逆映射檢查在最小奇異值小於 `1e-10`，或小於最大奇異值的 `1e-8` 時拒絕計算；此門檻以本工具的 SI 單位為基礎，不代表硬體可操作範圍的完整判定。

`diagnostics(q)` 可觀察條件數；接近奇異時，輸入誤差可能被放大。工具不會自動改用阻尼偽逆。

`ik`、`fk` 與行程轉換會檢查總長度限制；`geometry` 和衍生的 Jacobian／速度／力計算主要檢查角度及非零長度，不會全面檢查致動器行程。使用前可先呼叫 `w.ik(q)` 確認該姿態的長度可行性。工具也不會自動檢查速度、加速度或推力飽和。

## 10. 數值驗證範圍

`python wrist_tool.py --test` 包含兩組測試方法，內部覆蓋：

- 20 組固定亂數種子的姿態，驗證解析 Jacobian／Hessian 與中央有限差分一致。
- Hessian 對稱性、正逆位置／速度／加速度轉換。
- 沿時間軌跡的二階差分加速度檢查。
- 力映射的功率守恆、動力學正反轉換。
- 由旋轉矩陣微分檢查角速度，並檢查角加速度。
- 角度限制、不可達長度、步長拒絕、連續分支追蹤及奇異處拒絕。

這些驗證用於確認程式與模型公式的一致性，不代表已驗證實機幾何、感測器零位或負載參數。
