#! /usr/bin/env python

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from .progress import Progress

def requestsRetrySession(retries=3,backoff_factor=0.3,
                         status_forcelist=(500, 502, 504),
                         session=None):
    session = session or requests.Session()
    retry = Retry(total=retries,read=retries,connect=retries,
                  backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class DownloadFromGoogleDrive(object):

    @classmethod
    def download(cls,id,destination,**kwargs):
        URL = "https://docs.google.com/uc?export=download"
        session = requestsRetrySession(**kwargs)
        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = cls.get_confirm_token(response)        
        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        cls.save_response_content(response, destination)    
        return

    @classmethod
    def get_confirm_token(cls,response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @classmethod
    def save_response_content(cls,response,destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        return

