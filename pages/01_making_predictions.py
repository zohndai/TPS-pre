import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
#import cirpy
import torch
from rdkit.Chem import Draw
#import matplotlib.pyplot as plt
import os
import gdown
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

def smi_tokenize(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|su|OM|D|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|MW|u|v|r|l|E|US|heat|<|_)"
    compiled_pt = re.compile(pattern)
    tokens = [token for token in compiled_pt.findall(smi)]
    #assert smi == ''.join(tokens)
    return " ".join(tokens)
@st.cache_resource
def load_model(fd,model_name):
    file_id = fd
    model_path = model_name
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, model_path, quiet=True)
#if col1.button('Get the prediction')
@st.cache_data
def download():
	name = 'fine_tune_step_42480'
	destination_dir = 'models'
	os.makedirs(destination_dir, exist_ok=True)
	message_container = st.empty()
	message_container.text("Downloading the models... Please wait.")
	#https://drive.google.com/file/d/1MTK_uL2hyS2QJrmq4HjaVCdJNLkTf55s/view?usp=drive_link
	#1fa4ErVvXjZzbLCZzJEsQ6I9nelAR7iCu
	#1NmXn3OhaAexvfhdTwZQxq8gbKaF_GV6u
	fd_dict = {'1NmXn3OhaAexvfhdTwZQxq8gbKaF_GV6u':f'{name}_2024_0826'}
	for fd in fd_dict.keys():
		fd_file = fd
		model_name = fd_dict[fd]
		model_path = os.path.join(destination_dir, model_name)
		load_model(fd_file, model_path)
		time.sleep(4)
		current_file_path = f'models/{model_name}'
		new_file_path = f'models/{model_name}.pt'
		if not os.path.exists(new_file_path):
		    os.rename(current_file_path, new_file_path)
	message_container.text("Model is ready!")
	return new_file_path
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

st.set_page_config(
    page_title="Welcome to TP-Transformer",    
    page_icon="log.ico",        
    layout="wide",                
    initial_sidebar_state="auto"
)


def run():
	ros_name = ["HO∙", "SO₄∙⁻","O₃", "¹O₂",  "Fe(VI)", "O₂∙⁻", "MnO₄⁻", "ClO⁻","HClO", "Cl₂","Cl∙","CO₃∙⁻","Cl₂∙⁻","C₂H₃O₃∙", \
             "Cu(III)","Fe(V)",  "NO₂∙", "Mn(V)", "HSO₄∙", "O₂", "BrO⁻","NO∙", "ClO∙","Fe(IV)","Br∙", "IO⁻","C₂H₃O₂∙",\
             "HSO₅⁻", "ClO₂∙", "Br₂","HOBr","HO₂⁻","I∙", "NO₃∙", "IO₃∙⁻", \
           "Fe(III)", "S₂O₈∙⁻","HCO₃∙", "SO₃∙⁻","Unkown"]
	ros_smis = ['[OH]','[O]S(=O)(=O)[O-]','O=[O+][O-]','OO1','O=[Fe](=O)([O-])[O-]','[O][O-]','O=[Mn](=O)(=O)[O-]','[O-]Cl','OCl','ClCl','[Cl]','[O]C(=O)[O-]','Cl[Cl-]',\
	 'CC(=O)O[O]','[Cu+3]','O=[Fe]([O-])([O-])[O-]','[O]N=O','O=[Mn]([O-])([O-])[O-]','[O]S(=O)(=O)O','O=O','[O-]Br','[N]=O','[O]Cl','[O-][Fe]([O-])([O-])[O-]','[Br]',\
	 '[O-]I','CC([O])=O','O=S(=O)([O-])OO','[O][Cl+][O-]','BrBr','OBr','[O-]O','[I]','[O][N+](=O)[O-]','[O-][I+2]([O-])[O-]','[Fe+3]','[O]S(=O)(=O)OOS(=O)(=O)[O-]',\
	 '[O]C(=O)O','[O]S(=O)[O-]','']
	
	acti_methd=["UV light", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight", "No energy input"]
	methd_tokens=["UV", "heat", "VL", "MW", "E", "US", "hv", ""]
	
	st.subheader('What pollutant?')
	poll = st.text_input("Please input the SMILES notation for the pollutant, e.g. 'c1ccccc1' for benzene", "c1ccccc1")
	with st.expander("Show how to get SMILES of chemicals"):
		st.write('You can get SMILES of any molecules from PubChem https://pubchem.ncbi.nlm.nih.gov/ by typing Chemical name or ACS number')
	if poll =='':
		st.warning('Provide at least one molecular compound.')
		st.stop()
	
	st.subheader('Please select the ROSs that drive the pollutant degradation')
	ros_selct=st.selectbox('What ROSs? If not sure, select "Unknown"', ( "HO∙", "SO₄∙⁻","O₃", "¹O₂",  "Fe(VI)", "O₂∙⁻", "MnO₄⁻", "ClO⁻","HClO", "Cl₂","Cl∙","CO₃∙⁻","Cl₂∙⁻","C₂H₃O₃∙", \
	             "Cu(III)","Fe(V)",  "NO₂∙", "Mn(V)", "HSO₄∙", "O₂", "BrO⁻","NO∙", "ClO∙","Fe(IV)","Br∙", "IO⁻","C₂H₃O₂∙",\
	             "HSO₅⁻", "ClO₂∙", "Br₂","HOBr","HO₂⁻","I∙", "NO₃∙", "IO₃∙⁻", \
	           "Fe(III)", "S₂O₈∙⁻","HCO₃∙", "SO₃∙⁻", "Unkown"))
	#st.write('You selected:', ros_selct)
	#select = st.radio("Please specify the property or activity you want to predict", ('OH radical', 'SO4- radical', 'Koc', 'Solubility','pKd','pIC50','CCSM_H','CCSM_Na', 'Lipo','FreeSolv' ))
	st.subheader('Which precursors generate ROSs')
	prec = st.text_input("Please enter the SMILES notation for the precursor(s), including the parent oxidant and any activator (if applicable), e.g. 'OO.[Fe+2]' for the fenton reagent H2O2/Fe2+, if not sure, leave this field blank", "OO.[Fe+2]")
	#if prec !='':
		#st.warning('Invalid chemical name or CAS number of precursors, please check it again or imput SMILES')
		#st.stop()
	
	st.subheader("What energy input")
	methd_selct=st.selectbox("Please select the input energy for the ROSs generation",("UV light", "Heat", "Visible light", "Microwave", "Electricity", "Ultrasound", "Sunlight", "No energy input"),7)
	
	# st.subheader('Please input the reaction pH for pollutant degradation')
	# pH_value = st.text_input("Keep two decimal places","3.00")

	

# Display slider for general selection
#	pH_value = st.select_slider(
#	    'Select a value:',
#	    options=[round(x * 0.01, 2) for x in range(000, 1401)],
#	    value=st.session_state.value
#)
	
	# "+" and "-" buttons for fine-tuning
	

	
	#pH_value = "%.2f"%(st.select_slider('Select a pH value:',options=[round(x * 0.01, 2) for x in range(0000, 1401)], value=3.00))
	
	pH_value = "{:.2f}".format(st.select_slider('Select a pH value:',options=[round(x * 0.01, 2) for x in range(0000, 1401)], value=3.00))
	st.write('Selected pH value:', pH_value)

	col1, col2, col3, col4= st.columns([2,2,1,1])
	ros_smi = ros_smis[ros_name.index(ros_selct)]
	methd_token = methd_tokens[acti_methd.index(methd_selct)]
	pH = "".join(str(pH_value).split("."))
	try:
		cano_prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec))
	except:
		st.warning("invalid precursors's SMILES, please check it again")
		st.stop()
	
	try:
		cano_pollu = Chem.MolToSmiles(Chem.MolFromSmiles(poll))
	except:
		st.warning("invalid pollutant SMILES, please check it again")
		st.stop()
	reactant = cano_pollu + "." + ros_smi
	
	src = reactant+">"+cano_prec+"<"+methd_token+"_"+pH
	input = smi_tokenize(src)
	with open("src.txt", "w") as file:
		file.write(input)
	
	if col1.button('Get the prediction'):
		if all([not(prec), not(ros_smi)]):
			st.warning("At least one of 'ROSs' and 'precursors' should be given, please check your input again")
			st.stop()
		model_path = download()
		message_container = st.empty()
		message_container.text("model version:TP-Transformer-1.0.20240826")
	
		parser_tsl = ArgumentParser(description="translate.py")
		opts.config_opts(parser_tsl)
		opts.translate_opts(parser_tsl)
		args_tsl = ['-model', model_path, \
			    '-src', 'src.txt', \
			    '-output', 'predictions.txt', \
			    '-n_best', '10', \
			    '-beam_size', '10', \
			    '-max_length', '3000', \
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
		#message = "This is your message tip!"
		#background_color = "#00FF00"  # Green color
		
		# Use st.markdown to display the message with custom background color

		
		dp_smis = pd.read_csv(opt_tsl.output,header=None)
		smis_li=[".".join(list(set(("".join(dp_smi.split(" "))).split(".")))) for dp_smi in dp_smis[0]]
		if len(smis_li) != 10:
			smis_li += [""] * (10 - len(smis_li))
		recurr_list = []
		for i in range(10):
			list_cache = set(recurr_list)
			smils_i = smis_li[i].split(".")
			smis_li[i] = ".".join([smiles for smiles in smils_i if smiles not in list_cache])
			recurr_list += smils_i
		message_container = st.empty()
		
		message_container.text(",".join([f"top{i}:{smis_li[i-1]}" for i in range(1,11)]))
		Fig1_col,Fig2_col,Fig3_col,Fig4_col,Fig5_col, Fig6_col, Fig7_col,Fig8_col,Fig9_col,Fig10_col, = st.columns([1]*10)
		for i in range(1,11):
			try:
				cano_pro = Chem.MolToSmiles(Chem.MolFromSmiles(smis_li[i-1]))
				exec(f"top{i}_fig = Draw.MolToImage(Chem.MolFromSmiles(smis_li[i-1]))")
				eval(f"Fig{i}_col").image(eval(f"top{i}_fig"), caption = f'top{i}')
			except:
				eval(f"Fig{i}_col").image(Image.open("invalsmi.jpg"), caption = f'top{i}')
			st.cache_data.clear()
			st.cache_resource.clear()
	return

if __name__ == "__main__":
	run()

