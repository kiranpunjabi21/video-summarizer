from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import sklearn
import re
import logging
from helper import get_transcript
from model import loadBART
import os
import gtts
import torch
from fastapi import FastAPI
import playsound
import uvicorn
from fastapi.responses import FileResponse
nltk.download('stopwords')



app = FastAPI()

def get_tokens(link):

    try:
        tokenizer,model  = loadBART()
    except Exception as e:
        print("Not able to load BART model and BART tokenizer")
        logging.exception(f'Model not found due to {e}')
    
    
    full_transcript = get_transcript(link)

    if full_transcript is not None:
            
        tokens = tokenizer.encode(full_transcript,return_tensors= 'pt',max_length=512)
        output_tensor = model.generate(tokens,max_new_tokens=512)
        summary = tokenizer.decode(output_tensor[0])
        print('Summary is ',summary)
        return summary
    
    else:
        logging.exception('Not able to generate Summary')
        return None



def convert_summary_to_speech(summary):
    speech_summ = gtts.gTTS(summary)
    return speech_summ


@app.get("/generated-speech")
async def main(link:str):

    logging.basicConfig(filename= 'app.log',level=logging.INFO)
    youtube_link = link 
    summary = get_tokens(youtube_link)
    print("Summary is ",summary)
    speech_summ = convert_summary_to_speech(summary)
    unique_id = link.split('=')[-1]
    speech_summ.save(f'{unique_id}_summary.mp3')
    #playsound.playsound(speech_summ)
    print("Summary saved successfully")
    return FileResponse(f'{unique_id}_summary.mp3',media_type ="audio/mpeg")



if __name__ == '__main__':
    mp3_file = main('https://www.youtube.com/watch?v=Ic0TBhfuOrA')
    uvicorn.run(app,port=8000)
    print(f'MP3 File path {mp3_file}')


















