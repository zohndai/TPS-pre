import streamlit as st
import pandas as pd
#from numpy import np
from PIL import Image
import streamlit.components.v1 as components
import base64
from io import BytesIO

def create_contact_logo():
    # 创建画布
    from PIL import Image, ImageDraw, ImageFont
    width, height = 300, 150
    img = Image.new('RGB', (width, height), (25, 39, 52))  # 深蓝背景
    draw = ImageDraw.Draw(img)
    
    # 添加图标和文字
    try:
        # 尝试加载字体 (使用Streamlit内置字体)
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 绘制联系图标
    draw.ellipse([(30, 30), (80, 80)], outline="#4FC3F7", width=3)  # 头部
    draw.line([(55, 80), (55, 120)], fill="#4FC3F7", width=3)       # 身体
    draw.line([(30, 100), (80, 100)], fill="#4FC3F7", width=3)      # 手臂
    
    # 添加联系信息
    draw.text((100, 40), "遇到问题?", font=font, fill="#E3F2FD")
    draw.text((100, 70), "联系开发者:", font=font, fill="#4FC3F7")
    draw.text((100, 100), "contact@yourdomain.com", font=font, fill="#FFFFFF")
    
    return img



st.set_page_config(
    page_title="Welcome to TP-Transformer",    
    page_icon="log.ico",        
    layout="wide",                
    initial_sidebar_state="auto" 
)


visitor = pd.read_csv("visi_num.txt")
visi_num = visitor['num'][0]

if 'visitor_count' not in st.session_state:
	st.session_state.visitor_count = int(visi_num)
if 'session_initialized' not in st.session_state:
	st.session_state.session_initialized = True
st.session_state.visitor_count += 1
visitor['num'][0] += 1
st.metric(label=f'👀page views: {visi_num}', value='')
visitor.to_csv("visi_num.txt", index=False)

TEXT1 = """
        <body style='text-align: justify; color: black;'>
        <p>✨The TP-Transformer platform is powered by advanced machine learning models to assist users in predicting the transformation products of aqueous organic 
	pollutants in chemical oxidation processes. TP-Transformer is now capable of predicting both the degradation products and pathways of organic pollutants. It 
 	utilizes SMILES notation to represent chemical structures.   
        </p>🌟TP-Transformer is built on a Transformer architecture. It accepts pollutant SMILES, oxidative species, and reaction conditions (e.g., pH) as inputs, 
	and outputs the SMILES of the degradation products. This model can predict not only degradation intermediates but also complete degradation pathways. The 
 prediction of degradation pathways is achieved through an iterative process, where the degradation product predicted by TP-Transformer is used as input for subsequent 
 predictions. This process continues until the model predicts CO<sub>2</sub> or when the predicted chemicals remain unchanged (i.e., non-degradable), indicating the 
 formation of the final degradation products (Figure 1).
	  <p> 
	  </p>
        </body>         
        """

#
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

st.header('🎈Welcome to TP-Transformer!🎉')
st.markdown(f'{TEXT1}', unsafe_allow_html=True)
st.image(Image.open('predic.jpg'), width=1400, caption = 'Figure 1. The workflow that TP-Transformer makes predictions')


particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1;
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content"></div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false
        },
        "size": {
          "value": 2,
          "random": true
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#f6dcfa",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""


# 将联系方式转换为Base64
contact_logo = create_contact_logo()
buffered = BytesIO()
contact_logo.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# 使用HTML/CSS创建固定位置的联系按钮
st.markdown(
    f"""
    <style>
    .contact-badge {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s;
        cursor: pointer;
        background: rgba(25, 39, 52, 0.9);
        padding: 5px;
        max-width: 120px;
    }}
    .contact-badge:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }}
    </style>
    
    <div class="contact-badge" onclick="window.location.href='mailto:contact@yourdomain.com';">
        <img src="data:image/png;base64,{img_str}" alt="联系我们" style="width:100%; border-radius:6px;">
    </div>
    """,
    unsafe_allow_html=True
)

# 悬停提示
st.markdown(
    """
    <script>
    // 添加悬停提示
    document.addEventListener('DOMContentLoaded', function() {
        const badge = document.querySelector('.contact-badge');
        badge.title = "点击联系开发者解决问题";
    });
    </script>
    """,
    unsafe_allow_html=True
)



if "has_snowed" not in st.session_state:
    st.snow()
    st.session_state["has_snowed"] = True

if st.session_state.show_animation:
    components.html(particles_js, height=400, scrolling=False)
