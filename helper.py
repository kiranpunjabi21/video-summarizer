from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import sklearn
import re
from transformers import BartTokenizer, BartForConditionalGeneration,pipeline
import logging

def get_transcript(link):

    try:
        unique_id = link.split('=')[-1]
        transcript = YouTubeTranscriptApi.get_transcript(unique_id)
        full_transcript = " ".join(x['text'] for x in transcript)
        print("Full transcript is ",full_transcript)
        logging.info('Transcription generated successfully')
        return full_transcript

    except Exception as e:
        logging.exception('Not able to generate transcript due to {e}')
        return None

    