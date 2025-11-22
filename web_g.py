import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語ラベル用（不要ならコメントアウト）
import streamlit as st

# ===== 数値計算部分 =====

def simulate_trajectory(theta_degree, k=0.0, v0=1.0, m=1.0, g=9.8,
                        delta_t=1e-4, max_time=10.0):
    """
    角度 theta_degree [deg] でボールを投げたときの軌道をオイラー法で計算
    返り値: (xx_ary, yy_ary, t_ary)
    """
    theta = math.radians(theta_degree)

    # 速度ベクトル
    v = np.array([v0 * math.cos(theta), v0 * math.sin(theta)], dtype=float)
    # 位置ベクトル
    x = np.zeros(2, dtype=float)

    t = 0.0
    xx_list = [x[0]]
    yy_list = [x[1]]
    t_list  = [t]

    # 安全のためステップ上限
    max_steps = int(max_time / delta_t)

    for _ in range(max_steps):
        # 地面に落ちたら終了
        if x[1] < 0 and t > 0:
            break

        # 加速度 a = (-m*g - k*v) / m
        a = (-m * np.array([0.0, g]) - k * v) / m

        # オイラー法
        v = v + a * delta_t
        x = x + v * delta_t
        t += delta_t

        xx_list.append(x[0])
        yy_list.append(x[1])
        t_list.append(t)

    return np.array(xx_list), np.array(yy_list), np.array(t_list)


def range_of_theta(theta_degree, k=0.0, v0=1.0, m=1.0, g=9.8,
                   delta_t=1e-4):
    """指定角度で投げたときの飛距離（x の最後の値）"""
    xx, yy, t = simulate_trajectory(theta_degree, k=k, v0=v0, m=m,
                                    g=g, delta_t=delta_t)
    return float(xx[-1])


def gradient_descent_opt_angle(theta_init=30.0, k=0.0, v0=1.0, m=1.0, g=9.8,
                               delta_t=1e-4, delta_degree_theta=1.0,
                               alpha=10.0, num_iter=50):
    """
    球の飛距離の逆数を目的関数にして最急降下法で
    「一番飛ぶ角度」を探索する（元記事と同じアイデア）
    """
    theta = theta_init
    theta_history = []
    iter_history = []

    for i in range(num_iter):
        # 数値微分: d/dθ [ 1 / range(θ) ]
        theta1 = theta
        L1 = range_of_theta(theta1, k=k, v0=v0, m=m, g=g, delta_t=delta_t)
        # 0 除算ガード
        if L1 <= 0:
            break
        f1 = 1.0 / L1

        theta2 = theta + delta_degree_theta
        L2 = range_of_theta(theta2, k=k, v0=v0, m=m, g=g, delta_t=delta_t)
        if L2 <= 0:
            break
        f2 = 1.0 / L2

        df_dtheta = (f2 - f1) / delta_degree_theta

        # 勾配降下ステップ
        theta = theta - alpha * df_dtheta

        theta_history.append(theta)
        iter_history.append(i)

    return np.array(iter_history), np.array(theta_history), theta


# ===== Streamlit アプリ部分 =====

st.title("ボールを最も遠くに飛ばせる条件シミュレーション（Streamlit版）")
st.write("Qiita 記事のオイラー法シミュレーションを Streamlit でWebアプリ化したものです。")
st.page_link("https://qiita.com/arairuca/items/265f4008e2a5899d6eae", label="元ネタ記事はこちら")
# サイドバーでパラメータ設定
st.sidebar.header("シミュレーション設定")

mode = st.sidebar.radio(
    "モード選択",
    ["軌道を描く", "最急降下法で最適角度を探す"]
)

v0 = st.sidebar.slider("初速度 v₀ [m/s]", 0.1, 50.0, 20.0, 0.1)
k  = st.sidebar.slider("空気抵抗係数 k", 0.0, 20.0, 0.0, 1.0)
m  = st.sidebar.slider("ボールの質量 m [kg]", 0.1, 5.0, 1.0, 0.1)
g  = st.sidebar.slider("重力加速度 g [m/s²]", 1.0, 20.0, 9.8, 0.1)
delta_t = st.sidebar.select_slider(
    "時間刻み Δt（小さいほど精度↑・処理重い）",
    options=[1e-3, 5e-4, 1e-4],
    value=1e-4,
    format_func=lambda x: f"{x:.0e}"
)

st.sidebar.caption("※元記事では Δt = 1e-5 だが、Webアプリなのでやや粗めにしています。")

if mode == "軌道を描く":
    st.subheader("ボールの軌道シミュレーション")

    theta_degree = st.slider("投げる角度 θ [度]", 0.0, 90.0, 45.0, 0.5)

    if st.button("シミュレーション実行"):
        xx, yy, t = simulate_trajectory(theta_degree, k=k, v0=v0, m=m,
                                        g=g, delta_t=delta_t)

        # グラフ描画
        fig, ax = plt.subplots()
        # 投げる方向の直線（理想の初期方向）
        xx_line = np.linspace(0, max(xx), 100)
        theta_rad = math.radians(theta_degree)
        yy_line = math.tan(theta_rad) * xx_line
        ax.plot(xx_line, yy_line, linestyle=":", label="投げる方向")

        # ボールの軌道
        ax.plot(xx, yy, label="ボールの軌道")

        ax.set_xlabel("x軸 [m]")
        ax.set_ylabel("y軸 [m]")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        st.write(f"**飛距離（着地点 x） ≒ {xx[-1]:.3f} [m]**")
        st.write(f"シミュレーションステップ数: {len(t)}")

elif mode == "最急降下法で最適角度を探す":
    st.subheader("最急降下法による『一番飛ぶ角度』探索")

    theta_init = st.slider("初期角度 θ₀ [度]", 0.0, 80.0, 30.0, 1.0)
    num_iter   = st.slider("試行回数（反復回数）", 1, 100, 50, 1)
    alpha      = st.slider("学習率 α（大きいほど変化が激しい）", 0.1, 50.0, 10.0, 0.1)
    delta_theta = st.slider("数値微分の刻み Δθ [度]", 0.1, 5.0, 1.0, 0.1)

    st.caption("元記事と同様、目的関数は『飛距離の逆数』です。")

    if st.button("最急降下法を実行"):
        iters, thetas, theta_opt = gradient_descent_opt_angle(
            theta_init=theta_init,
            k=k,
            v0=v0,
            m=m,
            g=g,
            delta_t=delta_t,
            delta_degree_theta=delta_theta,
            alpha=alpha,
            num_iter=num_iter
        )

        if len(thetas) == 0:
            st.error("計算がうまく収束しませんでした。パラメータ（α や Δθ、Δt）を変えて再試行してみてください。")
        else:
            # 角度の収束の様子をプロット
            fig, ax = plt.subplots()
            ax.plot(iters, thetas, marker="o")
            ax.set_xlabel("試行回数")
            ax.set_ylabel("ボールを投げる角度 [度]")
            ax.grid(True)
            st.pyplot(fig)

            st.write(f"**推定された最適角度 θ ≒ {theta_opt:.2f} [度]**")

            # その角度で実際の軌道も描いてみる
            st.markdown("### 推定された最適角度での軌道")
            xx_opt, yy_opt, t_opt = simulate_trajectory(
                theta_opt, k=k, v0=v0, m=m, g=g, delta_t=delta_t
            )
            fig2, ax2 = plt.subplots()
            ax2.plot(xx_opt, yy_opt)
            ax2.set_xlabel("x軸 [m]")
            ax2.set_ylabel("y軸 [m]")
            ax2.grid(True)
            st.pyplot(fig2)

            st.write(f"**このときの飛距離 ≒ {xx_opt[-1]:.3f} [m]**")
