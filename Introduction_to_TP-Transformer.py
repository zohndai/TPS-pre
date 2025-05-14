import streamlit as st

# 设置页面配置
st.set_page_config(page_title="分子降解路径预测平台", layout="wide")

# 注入粒子背景 HTML + JS
particles_background = """
<style>
    #particles-js {
        position: fixed;
        width: 100%;
        height: 100%;
        background-color: #0d47a1;
        background-size: cover;
        background-position: 50% 50%;
        z-index: -1;
        top: 0;
        left: 0;
    }
</style>

<div id="particles-js"></div>

<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>

<script>
particlesJS("particles-js", {
    particles: {
      number: {
        value: 80,
        density: {
          enable: true,
          value_area: 800
        }
      },
      color: { value: "#ffffff" },
      shape: {
        type: "circle",
        stroke: { width: 0, color: "#000000" },
        polygon: { nb_sides: 5 }
      },
      opacity: {
        value: 0.5,
        random: false,
        anim: { enable: false, speed: 1, opacity_min: 0.1, sync: false }
      },
      size: {
        value: 3,
        random: true,
        anim: { enable: false, speed: 40, size_min: 0.1, sync: false }
      },
      line_linked: {
        enable: true,
        distance: 150,
        color: "#ffffff",
        opacity: 0.4,
        width: 1
      },
      move: {
        enable: true,
        speed: 6,
        direction: "none",
        random: false,
        straight: false,
        out_mode: "out",
        bounce: false,
        attract: { enable: false, rotateX: 600, rotateY: 1200 }
      }
    },
    interactivity: {
      detect_on: "canvas",
      events: {
        onhover: { enable: true, mode: "repulse" },
        onclick: { enable: true, mode: "push" },
        resize: true
      },
      modes: {
        grab: {
          distance: 400,
          line_linked: { opacity: 1 }
        },
        bubble: {
          distance: 400,
          size: 40,
          duration: 2,
          opacity: 8,
          speed: 3
        },
        repulse: {
          distance: 200,
          duration: 0.4
        },
        push: {
          particles_nb: 4
        },
        remove: {
          particles_nb: 2
        }
      }
    },
    retina_detect: true
  });
</script>
"""

st.markdown(particles_background, unsafe_allow_html=True)

# Streamlit 主体内容
st.title("🌟 分子降解路径预测平台")
st.markdown("欢迎使用本平台，通过输入前体信息，预测其在不同活性氧物种条件下的降解路径。")

species = st.selectbox("选择活性物种（ROS）", ["·OH", "SO₄·⁻", "O₃"])
precursor = st.text_input("请输入前体分子（IUPAC名称或SMILES）")

if st.button("开始预测"):
    if precursor:
        st.success(f"使用 {species} 预测 `{precursor}` 的降解路径：\n\n👉 模型预测结果展示在此")
    else:
        st.warning("请先输入前体分子结构！")
