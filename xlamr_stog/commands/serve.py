import tempfile
from typing import List, Iterator, Optional
import argparse
import sys
import json
import os
import torch
from tqdm.auto import tqdm

from xlamr_stog.data.dataset_readers.amr_parsing.amr import AMR
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.data.dataset_readers.amr_parsing.preprocess.input_cleaner import clean
from xlamr_stog.data.dataset_readers.amr_parsing.preprocess.sense_remover import SenseRemover
from xlamr_stog.data.dataset_readers.amr_parsing.preprocess.text_anonymizor import TextAnonymizor, load_name_bn_wiki_map, load_name_span_map
from xlamr_stog.utils.checks import check_for_gpu, ConfigurationError
from xlamr_stog.utils import lazy_groups_of
from xlamr_stog.predictors.predictor import Predictor, JsonDict
from xlamr_stog.predictors import STOGPredictor
from xlamr_stog.data import Instance
from xlamr_stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Use a trained model to make predictions.')
    parser.add_argument('--archive-file', required=True, type=str, help='the archived model to make predictions with')
    parser.add_argument('--weights-file', type=str, help='a path that overrides which weights file to use')
    parser.add_argument('--util-dir', required=True, type=str, help='The directory containing utils')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('--lang', required=True, type=str, help='language code')
    parser.add_argument('--predictor',
                           type=str,
                           help='optionally specify a specific predictor to use')
    parser.add_argument('--beam-size',
                        type=int,
                        default=1,
                        help="Beam size for seq2seq decoding")
    parser.add_argument('--raw-text',
                        type=str,
                        default="",
                        help="Raw Text")

    args = parser.parse_args()
    predictor = _get_predictor(args)

    if args.cuda_device >= 0:
        device = torch.device('cuda:{}'.format(args.cuda_device))
    else:
        device = torch.device('cpu')
    args.cuda_device = device
    fname = _write_to_file(args)
    text_anonymizer = _get_anonymizer(args)
    node_utils = NU.from_json(args.util_dir, 0)
    remover = SenseRemover(node_utils)
    for amr in AMRIO.read(fname):
        clean(amr)
        amr.abstract_map = text_anonymizer(amr)
        remover.remove_graph(amr)
        res = _predict(predictor, args, amr)
    print("Result:\n")
    print(res)    

