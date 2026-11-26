"""
Yay-Kurə Sisteminin Sönən Rəqslərinin Simulyasiyası
----------------------------------------------------
Bu proqram bir kürənin yaya bağlı sönən rəqslərini
həm animasiya, həm də yerdəyişmə-zaman qrafiki ilə göstərir.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================================
# 1) PARAMETRLƏR (bunları dəyişərək fərqli nəticələr görə bilərsən)
# ==========================================================
m = 1.0        # kütlə (kg)
k = 50.0       # yay sərtliyi (N/m)
c = 0.5        # sönmə əmsalı (N*s/m) -- 0 qoysan, sönməyən rəqs olar

x0 = 1.0       # başlanğıc yerdəyişmə (m)
v0 = 0.0       # başlanğıc sürət (m/s)

t_max = 4.0   # simulyasiyanın neçə saniyə davam edəcəyi
dt = 0.02      # zaman addımı (nə qədər kiçikdirsə, bir o qədər dəqiqdir)

# ==========================================================
# 2) HƏRƏKƏT TƏNLİYİNİN HƏLLİ (Runge-Kutta 4-cü dərəcə üsulu)
# ==========================================================
# Tənlik: m*x'' + c*x' + k*x = 0
# Bunu iki tənliyə bölürük: x' = v,  v' = -(c*v + k*x)/m

def acceleration(x, v):
    return -(c * v + k * x) / m

n_steps = int(t_max / dt)
t_values = np.zeros(n_steps)
x_values = np.zeros(n_steps)

x, v = x0, v0
for i in range(n_steps):
    t_values[i] = i * dt
    x_values[i] = x

    # Runge-Kutta 4 addımı
    k1x, k1v = v, acceleration(x, v)
    k2x, k2v = v + 0.5*dt*k1v, acceleration(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x, k3v = v + 0.5*dt*k2v, acceleration(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x, k4v = v + dt*k3v,     acceleration(x + dt*k3x,     v + dt*k3v)

    x += (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
    v += (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)

# ==========================================================
# 3) VİZUAL HİSSƏ: iki panel - animasiya və qrafik
# ==========================================================
fig, (ax_anim, ax_graph) = plt.subplots(1, 2, figsize=(10, 5))

# --- Sol panel: yay + kürə animasiyası ---
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(0, 1.5)
ax_anim.set_title("Yay-Kürə Sistemi")
ax_anim.set_xticks([])
ax_anim.set_yticks([])

spring_line, = ax_anim.plot([], [], lw=2, color="gray")
mass_point, = ax_anim.plot([], [], "o", markersize=25, color="royalblue")
wall_y = 0.75  # yayın bağlandığı nöqtənin hündürlüyü

# --- Sağ panel: yerdəyişmə-zaman qrafiki ---
ax_graph.set_xlim(0, t_max)
ax_graph.set_ylim(min(x_values)*1.2 - 0.1, max(x_values)*1.2 + 0.1)
ax_graph.set_title("Yerdəyişmə - Zaman Qrafiki")
ax_graph.set_xlabel("Zaman (s)")
ax_graph.set_ylabel("Yerdəyişmə x (m)")
ax_graph.grid(True, alpha=0.3)
    
graph_line, = ax_graph.plot([], [], color="crimson", lw=2)

def init():
    spring_line.set_data([], [])
    mass_point.set_data([], [])
    graph_line.set_data([], [])
    return spring_line, mass_point, graph_line

def make_spring_shape(x_pos, n_coils=8):
    """Sadə zigzaq formasında yay xətti yaradır."""
    xs = np.linspace(-1.8, x_pos, n_coils * 2)
    ys = wall_y + 0.05 * (-1) ** np.arange(len(xs))
    ys[0] = wall_y
    ys[-1] = wall_y
    return xs, ys

def update(frame):
    current_x = x_values[frame]

    # Yay və kürənin görünüşünü yenilə
    xs, ys = make_spring_shape(current_x)
    spring_line.set_data(xs, ys)
    mass_point.set_data([current_x], [wall_y])

    # Qrafiki tədricən çək
    graph_line.set_data(t_values[:frame], x_values[:frame])

    return spring_line, mass_point, graph_line

ani = animation.FuncAnimation(
    fig, update, frames=n_steps, init_func=init,
    interval=dt*1000, blit=True, repeat=False
)

plt.tight_layout()
plt.show()