from transformers import BartTokenizer, BartForConditionalGeneration,pipeline
import logging

def loadBART():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model =  BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    return tokenizer, model