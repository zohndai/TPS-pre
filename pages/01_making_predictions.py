import streamlit as st
from streamlit_ketcher import st_ketcher
# import streamlit.components.v1 as components
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
#import cirpy
import torch
from rdkit.Chem import Draw
#import matplotlib.pyplot as plt
import os
# import gdown
from huggingface_hub import hf_hub_download
import time
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab, _load_vocab, DatasetLazyIter, OrderedIterator
from functools import partial
from multiprocessing import Pool
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab
import onmt.translate
import pandas as pd
import onmt.model_builder
import onmt.translate
from onmt.utils.misc import split_corpus
import re
from PIL import Image



model_collec = {"256":"fine_tune_step_49320_aop_plus_photo_best.pt", "512": "fine_tune_step_44460_best-512.pt"} 
model_ver_collec = {"256":"20250608-1.0.0", "512": "20250508-1.1.0"} 

def smi_tokenize(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|su|OM|D|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|MW|u|v|r|l|E|US|heat|<|_)"
    compiled_pt = re.compile(pattern)
    tokens = [token for token in compiled_pt.findall(smi)]
    #assert smi == ''.join(tokens)
    return " ".join(tokens)
def cano_smi(smi):
	if not smi:
		return smi
	smi_list = smi.split(".")
	smi_num = len(smi_list)
	for i in range(smi_num):
		try:
			smi_list[i] = Chem.MolToSmiles(Chem.MolFromSmiles(smi_list[i]))
		except:
			smi_list[i] = smi_list[i]		
	cano_smiles = ".".join(smi_list)
	return cano_smiles	      

@st.cache_data
def load_model(model_name, cache_dir="models"): 
    repo_name = "zohndai/tp_transformer-oxi-photo"
    model_path = hf_hub_download(repo_id=repo_name, filename=model_name, cache_dir=cache_dir)
    return model_path
	
# Streamlit 缓存 + 友好提示
@st.cache_data
def download(dimension):
	# name = 'Chem_Oxi_photo'
	destination_dir = 'models'
	os.makedirs(destination_dir, exist_ok=True)
	
	message_container = st.empty()
	message_container.text("⏳ Downloading the model... Please waite")


    # fd_dict = {
    #     'fine_tune_step_49320_aop_plus_photo_best.pt': f'{name}_2025_0508'
    # }

    # for remote_file, local_base_name in fd_dict.items():
        # 下载模型
	remote_file = model_collec[dimension]
	model_path = load_model(remote_file, cache_dir=destination_dir)
	time.sleep(0.1)

	message_container.text("🚀 Model is ready! ")
	return model_path
        # # 重命名下载的模型为自定义名称
        # current_file_path = model_path
        # new_file_path = os.path.join(destination_dir, local_base_name + ".pt")

        # if not os.path.exists(new_file_path):
        #     os.rename(current_file_path, new_file_path)


@st.cache_data	
def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])#加载模型参数
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):#False
        fields = inputters.load_old_vocab(vocab, opt.data_type, dynamic_dict=model_opt.copy_attn)
    else:
        fields = vocab
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, opt.gpu)
    if opt.fp32:#False
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt
def build_translator(opt, report_score, logger=None, out_file=None):
    log_probs_out_file = None
    target_score_out_file = None
    if out_file is None:#True
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
        
    if opt.log_probs:#False
        log_prob_oout_file = codecs.open(opt.ouput + '_log_probs', 'w+', 'utf-8')
        target_score_out_file = codecs.open(opt.output + '_gold_score', 'w+', 'utf-8')
        
    load_test_model = onmt.decoders.ensemble.load_test_model if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)
    
    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    
    translator = onmt.translate.Translator.from_opt(
        model, fields, opt, model_opt, global_scorer=scorer, out_file=out_file, report_align=opt.report_align, \
        report_score=report_score, logger=logger, log_probs_out_file=log_probs_out_file, target_score_out_file=target_score_out_file,
    )
    return translator

ros_name = ['HO∙','¹O₂','O₃','SO₄∙⁻','O₂∙⁻','3DOM*','MnO₄⁻','HOCl','Fe(VI)',\
	'Cl∙','ClO⁻','CO₃∙⁻','HFe(VI)','Cl₂','NO₂∙','Cl₂∙⁻','C₂H₃O₃∙','Cu(III)','C₃H₅O₂∙', \
	'NO∙','Fe(V)','Mn(III)', 'Fe(IV)','HSO₄∙','Mn(V)','ClO∙','O₂','BrO⁻',\
	'Cr₂O₇²⁻','Br∙','IO⁻','³OM*', 'C₂H₃O₂∙','HOBr','HSO₅⁻',\
	'IO₄⁻','Mn₂O₂','HNCl','Br₂','ClO₂∙','NO₃∙','I∙','HO₂⁻','HCO₃∙',\
	'S₂O₈∙⁻','SO₃∙⁻','IO₃⁻','Fe(III)','NO₂⁺','HOI', 'O', "Unkown"]
ros_smis = ['[OH]','1O=O','O=[O+][O-]','[O]S(=O)(=O)[O-]','[O][O-]','3DOM*','O=[Mn](=O)(=O)[O-]','OCl','O=[Fe](=O)([O-])[O-]',\
	'[Cl]','[O-]Cl','[O]C(=O)[O-]','O=[Fe](=O)([O-])O','ClCl','O=[N+][O-]','Cl[Cl-]','CC(=O)O[O]','[Cu+3]','CCC([O])=O', \
	'[N+][O-]','O=[Fe]([O-])([O-])[O-]','[Mn+3]', '[O-][Fe]([O-])([O-])[O-]','[O]S(=O)(=O)O','[Mn+5]','[O]Cl','O=O','[O-]Br',\
	'O=[Cr](=O)([O-])O[Cr](=O)(=O)[O-]','[Br]','[O-]I','3OM*', 'CC([O])=O','OBr','O=S(=O)([O-])OO',\
	'[O-][I+3]([O-])([O-])[O-]','[Mn+2].[Mn+2].[O-2].[O-2]','NCl','BrBr','[O][Cl+][O-]','[O][N+](=O)[O-]','[I]','[O-]O','[O]C(=O)O',\
	'[O]S(=O)(=O)OOS(=O)(=O)[O-]','[O]S(=O)[O-]','[O-][I+2]([O-])[O-]','[Fe+3]','O=[N+]=O','OI', 'O', '']
	
acti_methd=["UV light", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight", "Infrared", "No energy input"]
methd_tokens=["ul", "heat", "vl", "MW", "E", "US", "sul", "rl", ""]
default_thred = [
    0.911749519,  # top1
    0.363675185,  # top2
    0.300792199,  # top3
    0.336410101,  # top4
    0.257537316   # top5
]
oh_thred = [
    0.909504652999999,  # top1
    0.365850866,  # top2
    0.302991391999999,  # top3
    0.29491562,  # top4
    0.184767078   # top5
]

singlet_O_thred = [
    0.864012347,  # top1
    0.112246399,  # top2
    0.136595739,  # top3
    0.180508057,  # top4
    0.117356113   # top5
]

photo_thred = [
    0.920492263,  # top1
    0.429255432,  # top2
    0.45284178,  # top3
    0.154405293,  # top4
    0.305829397   # top5
]

def get_thred(ros, method):
	if ros == "HO∙":
		return oh_thred
	if ros == "¹O₂":
		return singlet_O_thred
	if all([ros in ['Unkown', 'O', '3DOM*','3OM*'], method in ["UV light", "Visible light", "Sunlight", "Infrared"]]):
		return photo_thred
	else:
		return default_thred
		
st.set_page_config(
    page_title="Welcome to TP-Transformer",    
    page_icon="log.ico",        
    layout="wide",                
    initial_sidebar_state="auto"
)

with st.sidebar:
	with st.expander("🛠️Specify prediction parameters"):
		model_select=st.selectbox("Select model dimension",("256", "512"), 1)

		st.subheader('💥Select ROSs that drive pollutant degradation')
		ros_selct=st.selectbox('What ROSs? If not sure, select "Unknown"', ('HO∙','¹O₂','O₃','SO₄∙⁻','O₂∙⁻','3DOM*','MnO₄⁻','HOCl','Fe(VI)',\
		'Cl∙','ClO⁻','CO₃∙⁻','HFe(VI)','Cl₂','NO₂∙','Cl₂∙⁻','C₂H₃O₃∙','Cu(III)','C₃H₅O₂∙', \
		'NO∙','Fe(V)','Mn(III)', 'Fe(IV)','HSO₄∙','Mn(V)','ClO∙','O₂','BrO⁻',\
		'Cr₂O₇²⁻','Br∙','IO⁻','³OM*', 'C₂H₃O₂∙','HOBr','HSO₅⁻',\
		'IO₄⁻','Mn₂O₂','HNCl','Br₂','ClO₂∙','NO₃∙','I∙','HO₂⁻','HCO₃∙',\
		'S₂O₈∙⁻','SO₃∙⁻','IO₃⁻','Fe(III)','NO₂⁺','HOI', 'O', "Unkown"))
		#st.write('You selected:', ros_selct)
		#select = st.radio("Please specify the property or activity you want to predict", ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv' ))
		st.subheader('🧪Which precursors generate ROSs')
		prec = st.text_input("Please enter the SMILES notation for the precursor(s), including the parent oxidant and any activator (if applicable), e.g. 'OO.[Fe+2]' for the fenton reagent H2O2/Fe2+, if not sure, leave this field blank", "")
		#if prec !='':
			#st.warning('Invalid chemical name or CAS number of precursors, please check it again or imput SMILES')
			#st.stop()
		
		st.subheader("⚡What energy input")
		methd_selct=st.selectbox("Please select the input energy for the ROSs generation",("UV light", "Heat", "Visible light", \
			       "Microwave", "Electricity", "Ultrasound", "Sunlight", "Infrared", "No energy input"),8)
		
		st.subheader('🌡️Enter the reaction pH')
		pH_value = st.text_input("Keep two decimal places","")
	
		
	
	# Display slider for general selection
	#	pH_value = st.select_slider(
	#	    'Select a value:',
	#	    options=[round(x * 0.01, 2) for x in range(000, 1401)],
	#	    value=st.session_state.value
	#)
		
		# "+" and "-" buttons for fine-tuning
	
		st.subheader('⚖️Specify probability thresholds')
		thed_cols1 = st.columns(3)
		thed_cols2 = st.columns(3)
	
		default_values = get_thred(ros_selct, methd_selct)
		
		thresholds = {}
		
		for i in range(5):
			thred_cols = thed_cols1 if i < 3 else thed_cols2
			j = i if i < 3 else i-3
			with thred_cols[j]:
				thresholds[i] = st.number_input(
				    label=f"top{i+1} threshold:",
				    min_value=0.0,
				    max_value=1.0,
				    value=default_values[i],
				    format="%.9f",
				    step=1e-9,
				    help=f"default: {default_values[i]:.6f}"
				)





	
	with st.expander("📞Contact developer"):
		
		st.markdown("📧 Email: [zhen.h.dai@outlook.com](mailto:zhen.h.dai@outlook.com)")
		# st.markdown("📱 Wechat: your_wechat_id")
		st.markdown("[📝 feedback](https://docs.qq.com/form/page/DVFdraEFYeEdCZEJ6)")


def run():
	st.subheader('🔬What pollutant?')
	default_mol = st.text_input("Please input the SMILES notation for the pollutant, e.g. 'c1ccccc1' for benzene", "c1ccccc1")
	with st.expander("📌Show how to get SMILES of chemicals"):
		st.write('You can get SMILES of any molecules from PubChem https://pubchem.ncbi.nlm.nih.gov/ by typing Chemical name or ACS number')
	# # text input for manually input molecular SMILES
	# molecule = st.text_input("molecular strcuture（SMILES）", default_smiles)
	# Ketcher  molecule editor
	st.markdown(f"✍Or manually draw the target pollutant below:")
	poll = st_ketcher(default_mol,height=500)
	# Showing molecule SMILES from editor
	st.markdown(f"**current SMILES：** `{poll}`")

	if poll =='':
		st.warning('⚠️Provide at least one molecular compound.')
		st.stop()
	
	col1, col2, col3, col4= st.columns([2,2,1,1])
	ros_smi = ros_smis[ros_name.index(ros_selct)]
	methd_token = methd_tokens[acti_methd.index(methd_selct)]
	pH = "".join(str(pH_value).split("."))
	try:
		cano_prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec))
	except:
		st.warning("⚠️invalid precursors's SMILES, please check it again")
		cano_prec = prec
		# st.stop()
	
	try:
		cano_pollu = Chem.MolToSmiles(Chem.MolFromSmiles(poll))
	except:
		st.warning("⚠️invalid pollutant SMILES, please check it again")
		st.stop()
	reactant = cano_pollu + "." + ros_smi
	
	src = reactant+">"+cano_prec+"<"+methd_token+"_"+pH
	input = smi_tokenize(src)
	with open("src.txt", "w") as file:
		file.write(input)
	
	if col1.button('Get the prediction'):
		# if all([not(prec), not(ros_smi)]):
		# 	st.warning("⚠️At least one of 'ROSs' and 'precursors' should be given, please check your input again")
		# 	st.stop()
		model_path = download(model_select)
		message_container = st.empty()
		message_container.text("🤖model version:TP-Transformer-%s"%(model_ver_collec[model_select] + "-dm-" + model_select))
	
		parser_tsl = ArgumentParser(description="translate.py")
		opts.config_opts(parser_tsl)
		opts.translate_opts(parser_tsl)
		args_tsl = ['-model', model_path, \
			    '-src', 'src.txt', \
			    '-output', 'predictions.txt', \
			    '-n_best', '5', \
			    '-beam_size', '10', \
			    '-max_length', '300', \
			    '-batch_size', '64']
		opt_tsl = parser_tsl.parse_args(args_tsl)
		ArgumentParser.validate_translate_opts(opt_tsl)
		#logg1er = init_logger(opt_tsl.log_file)
	#model_path = opt_tsl.models[0]
		checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
		vocab = checkpoint['vocab']
		translator = build_translator(opt_tsl, report_score=True)
		src_shards_tsl = split_corpus(opt_tsl.src, opt_tsl.shard_size) # list(islice(f, shard_size))
		tgt_shards_tsl = split_corpus(opt_tsl.tgt, opt_tsl.shard_size)
		shard_pairs_tsl = zip(src_shards_tsl, tgt_shards_tsl)
		for i, (src_shard_tsl, tgt_shard_tsl) in enumerate(shard_pairs_tsl): # 0, ([src], None)
			#logger.info("Translating shard %d." % i) # 
			translator.translate(
			    src=src_shard_tsl,
			    tgt=tgt_shard_tsl,
			    src_dir=opt_tsl.src_dir, #''
			    batch_size=opt_tsl.batch_size, # 64
			    batch_type=opt_tsl.batch_type, # sents
			    attn_debug=opt_tsl.attn_debug, # False
			    align_debug=opt_tsl.align_debug # False
			)

		
		st.success("Predicted Products:")


		
		dp_smis = pd.read_csv(opt_tsl.output,header=None)
		confid = pd.read_csv(
			opt_tsl.output.replace(".txt", "_confidence.csv"),
			header=None, names=["confidence"]
		)
		smis_li=[".".join(list(set(cano_smi("".join(dp_smi.split(" "))).split(".")))) for dp_smi in dp_smis[0]]
		if len(smis_li) != 5:
			smis_li += [""] * (5 - len(smis_li))
		# recurr_list = []
		for i in range(5):
			list_cache = list([cano_pollu])
		# 	smils_i = smis_li[i].split(".")
			smis_li[i] = ".".join([smiles for smiles in smis_li[i].split(".") if smiles not in list_cache])
		# 	recurr_list += smils_i

		
		message_container = st.empty()
		# message_container.markdown(f"{dp_smis[0]}")

		message_container.markdown("<br>".join([
			    f"<b>top{i}:</b> <span style='color:{
				    "#00ff00" if confid['confidence'][i-1] >= np.float64(thresholds[i-1]) else "#ff9900"
			    }; font-family: monospace;'>" +
			    smis_li[i-1].replace("[", "&#91;").replace("]", "&#93;") +  # 转义方括号
			    "</span>"
			    for i in range(1, 6)
			]), unsafe_allow_html=True)
		
		
		# message_container.markdown("<br>".join([
		# 	    f"<b>top{i}:</b> <span style='color:{
		# 		    "#00ff00" if confid['confidence'][i-1] >= np.float64(thresholds[i-1]) else "#ff9900"}; 
		# 			font-family: monospace;'>{smis_li[i-1]}</span>"
		# 	    		for i in range(1, 6)
		# 	]), unsafe_allow_html=True)
		 

		# st.markdown(",".join([f"**top{i}:** `{smis_li[i-1]}`" for i in range(1,11)])) 这行速度太慢了
		# message_container.text(",".join([f"**top{i}:** `{smis_li[i-1]}`" for i in range(1,11)]))

		# html_content = "<div>" + "<br>".join([
		#     f"<b>top{i}:</b> <span style='color:{"#00ff00" if confid['confidence'][i-1] >= np.float64(thresholds[i-1]) else "#ff9900"}; font-family: monospace;'>{smis_li[i-1]}</span>"
		#     for i in range(1, 6)
		# ]) + "</div>"
		
		# html(html_content)


		
		Fig1_col,Fig2_col,Fig3_col,Fig4_col,Fig5_col, = st.columns([1]*5)
		#Fig6_col, Fig7_col,Fig8_col,Fig9_col,Fig10_col, = st.columns([1]*10)
		conf1_col,conf2_col,conf3_col,conf4_col,conf5_col, = st.columns([1]*5)
		for i in range(1,6):
			try:
				cano_pro = Chem.MolToSmiles(Chem.MolFromSmiles(smis_li[i-1]))
				exec(f"top{i}_fig = Draw.MolToImage(Chem.MolFromSmiles(smis_li[i-1]))")
				eval(f"Fig{i}_col").image(eval(f"top{i}_fig"), caption = f'top{i}')
			except:
				eval(f"Fig{i}_col").image(Image.open("invalsmi.jpg"), caption = f'top{i}')
			# eval(f"conf{i}_col").text(f"confidence:{confid["confidence"][i-1]:5f}")
			color = "#00ff00" if confid['confidence'][i-1] >= np.float64(thresholds[i-1]) else "#ff9900"
			eval(f"conf{i}_col").markdown(
				f"""
			    <div style='
       				text-align: center;color:{color};font-family: monospace;'> confidence: {confid['confidence'][i-1]:.5f}
       				</div>
			    """,
			unsafe_allow_html=True
			)
			st.cache_data.clear()
			st.cache_resource.clear()
	return

if __name__ == "__main__":
	run()
