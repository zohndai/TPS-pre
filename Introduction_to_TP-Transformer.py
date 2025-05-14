import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="动态粒子背景", layout="wide")

# 粒子背景组件，确保 JS 在独立 iframe 中运行
html_code = """
<!DOCTYPE html>
<html>
  <head>
    <style>
      body, html {
        margin: 0;
        padding: 0;
        height: 100%;
        overflow: hidden;
        background-color: #0d47a1;
      }
      #particles-js {
        position: absolute;
        width: 100%;
        height: 100%;
        z-index: -1;
      }
    </style>
  </head>
  <body>
    <div id="particles-js"></div>
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>
      particlesJS("particles-js", {
        "particles": {
          "number": {
            "value": 60,
            "density": {
              "enable": true,
              "value_area": 800
            }
          },
          "color": {
            "value": "#ffffff"
          },
          "shape": {
            "type": "circle"
          },
          "opacity": {
            "value": 0.5,
            "random": false
          },
          "size": {
            "value": 3,
            "random": true
          },
          "line_linked": {
            "enable": true,
            "distance": 150,
            "color": "#ffffff",
            "opacity": 0.4,
            "width": 1
          },
          "move": {
            "enable": true,
            "speed": 2
          }
        },
        "interactivity": {
          "detect_on": "canvas",
          "events": {
            "onhover": {
              "enable": true,
              "mode": "repulse"
            }
          }
        },
        "retina_detect": true
      });
    </script>
  </body>
</html>
"""

# 注入 HTML
components.html(html_code, height=700)

# Streamlit 内容
st.title("🌟 分子降解路径预测平台")
st.markdown("上方是动态粒子背景，使用 `particles.js` 实现。")
