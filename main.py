from fastapi import FastAPI, Response
from dataclasses import dataclass
import tempfile
import json
import torch
from typing import List, Iterator, Optional
import argparse
import os
from tqdm.auto import tqdm

from xlamr_stog.data.dataset_readers.amr_parsing.amr import AMR
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner import clean
from xlamr_stog.data.dataset_readers.amr_parsing.preprocess.sense_remover import SenseRemover
from xlamr_stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor import TextAnonymizor, load_name_bn_wiki_map, load_name_span_map
from xlamr_stog.data.dataset_readers.amr_parsing.postprocess.wikification import Wikification
from xlamr_stog.data.dataset_readers.amr_parsing.postprocess.expander import Expander, convert_postags
from xlamr_stog.predictors.predictor import Predictor, JsonDict
from xlamr_stog.predictors import STOGPredictor
from xlamr_stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

@dataclass
class Arguments:
    archive_file:str
    weights_file:str
    util_dir:str
    raw_text:str
    cuda_device:int = 0
    beam_size:int=5
    lang:str="ms"
    predictor:str="STOG"
    exclude_spotlight:bool=True
    u_pos:bool=True

def _write_to_file(args):
    fname = ""
    raw_amr = args.raw_text.replace("\\n","\n")
    print(raw_amr)
    with tempfile.NamedTemporaryFile('w', delete=False) as fp:
        fp.write(raw_amr)
        fname = fp.name
    return fname

def _get_anonymizer(args):
    text_anonymizor = TextAnonymizor.from_json(os.path.join(args.util_dir,"text_anonymization_en-{}.json".format(args.lang)))
    text_anonymizor.lang_stopwords = set([x.rstrip() for x in open("data/cross-lingual-babelnet_mappings/stopwords_{}.txt".format(args.lang))])
    text_anonymizor.lang2en_span=load_name_span_map("data/cross-lingual-babelnet_mappings/name_span_en_{}_map_amr_bn.json".format(args.lang), args.lang)
    text_anonymizor.lang2en_bn=load_name_bn_wiki_map("data/cross-lingual-babelnet_mappings/namedEntity_wiki_synsets.{}.tsv".format(args.lang.upper()))
    
    return text_anonymizor

def _get_predictor(args: argparse.Namespace) -> Predictor:
    from xlamr_stog.utils.archival import load_archive
    # check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           device=args.cuda_device,
                           weights_file=args.weights_file)
    print("Loaded model weights.")
    return Predictor.from_archive(archive)


def _predict(predictor:Predictor, args: argparse.Namespace, amr:AMR) :
    predictor._dataset_reader.universal_postags = predictor._model.universal_postags
    predictor._dataset_reader.source_copy = predictor._model.generator_source_copy
    predictor._dataset_reader.translation_mapping = predictor._model.translation_mapping
    predictor._dataset_reader.multilingual = predictor._model.multilingual
    predictor._dataset_reader.extra_check = predictor._model.test_config.get('extra_check',False)
    
    if type(predictor) in (STOGPredictor,):
        predictor._model.set_beam_size(args.beam_size)
        predictor._model.set_decoder_token_indexers(predictor._dataset_reader._token_indexers)

    print("Reading Dataset")
    batch = predictor._dataset_reader.text_to_instance(
        amr,
        args.lang,
        predictor._dataset_reader.source_copy,
        predictor._dataset_reader.split )
    
    print("Predicting Result")
    pred_results = predictor.predict_batch_instance([batch])[0]
    return [predictor.dump_line(output) for output in pred_results]

def _restore_instance(amr, node_utils):
        graph = amr.graph
        for node in graph.get_nodes():
            instance = node.instance
            new_instance = node_utils.get_frames(instance)[0]
            if instance != new_instance:
                graph.replace_node_attribute(node, 'instance', instance, new_instance)
            continue

app = FastAPI()
model_dir="models/xl-amr_bilingual_en_ms_trans_amr_mt5"
args = Arguments(
    archive_file=model_dir,
    weights_file=os.path.join(model_dir,"best.th"),
    util_dir="data/AMR/en_ms_utils",
    raw_text="",
    cuda_device=-1
)

if args.cuda_device >= 0:
    device = torch.device('cuda:{}'.format(args.cuda_device))
else:
    device = torch.device('cpu')
args.cuda_device = device

predictor=_get_predictor(args)
node_utils = NU.from_json(args.util_dir, 0)
remover = SenseRemover(node_utils)
wikification = Wikification(util_dir=args.util_dir, lang=args.lang)
wikification.load_utils(exclude_spotlight=args.exclude_spotlight)
lang_vocab = json.load(open("data/numberbatch/{}_en_neighbors_model.json".format(args.lang), "r"))
postag_map = convert_postags(args.lang)
expander = Expander(util_dir=args.util_dir, lang_vocab=lang_vocab, postag_map=postag_map, lang =args.lang, u_pos=args.u_pos)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/full-predict")
async def full_predict(raw_text:str):
    args.raw_text=raw_text
    fname = _write_to_file(args)
    text_anonymizer = _get_anonymizer(args)
    for amr in AMRIO.read(fname, lang="ms"):
        if "#id" not in raw_text:
            amr.id = "0"
        clean(amr)
        amr.abstract_map = text_anonymizer(amr)
        remover.remove_graph(amr)
        res = _predict(predictor, args, amr)
        
    res_fname=""
    with tempfile.NamedTemporaryFile('w', delete=False) as fp:
        fp.write(str(res[0]))
        res_fname = fp.name
    
    for amr in AMRIO.read(res_fname, lang="ms"):
        try:
            _restore_instance(amr, node_utils)
        except Exception as e:
            print("error in processing instance")
            print(str(e))
        wikification.wikify_graph(amr, exclude_spotlight=args.exclude_spotlight)
        expander.expand_graph(amr)
        return Response(content=str(amr))