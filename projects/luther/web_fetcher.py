
# coding: utf-8

# # WEB Fetcher

# In[3]:


import sys
import os
import requests
import re
import gzip
import hashlib
import time
import random
from http import HTTPStatus
from urllib.parse import     urlparse
import multiprocessing as mp
import json


# In[2]:


# Cache dir location:
cache_dir = 'data/.web_fetcher'
# Number of buckets for each site cache:
num_buckets = 1000
# If a dir like URL is fecthed (path ends in '/') then use this suffix to
# store the results:
dir_url_suffix = '__dir'
# If path is null then use this file name to store the result:
null_path_replace = '__nopath'


# In[ ]:


def map_url_to_cache_file(url, cache_dir=cache_dir, num_buckets=num_buckets, 
                          dir_url_suffix=dir_url_suffix,
                          null_path_replace=null_path_replace):
    """Map URL to cache file path using site grouping and hashing
       to avoid too many files per directory.
    
    """
    
    # Save URL in:
    #   cache_dir/netloc/scheme[/port]/bucket_num/path
    
    bucket_num = '{:08x}'.format(
        int.from_bytes(hashlib.md5(url.encode()).digest(),
                       byteorder = sys.byteorder) % num_buckets
    )

    parsed_url = urlparse(url)

    url_netloc = parsed_url.netloc
    url_port = ''
    i = url_netloc.rfind(':')
    if i >= 0:
        url_port = url_netloc[i+1:]
        url_netloc = url_netloc[:i]

    url_path = parsed_url.path
    if url_path:
        if url_path[-1] == '/':
            url_path = url_path[:-1] + dir_url_suffix
        if url_path[0] == '/':
            url_path = url_path[1:]
    else:
        url_path = null_path_replace
 
    return os.path.join(cache_dir, 
                        url_netloc,
                        parsed_url.scheme,
                        url_port,
                        bucket_num, 
                        url_path)


# In[ ]:


def get_url(url, retries=3, pause_between_retries=10, timeout=30):
    
    for i in range(retries):
        if i > 0 and pause_between_retries > 0:
            time.sleep((1 + 05. * random.random()) * pause_between_retries)
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == HTTPStatus.OK:
                return response
            elif response.status_code in [HTTPStatus.FORBIDDEN]:
                continue
            else:
                break
        except requests.exceptions.Timeout:
            pass
    


# In[3]:


def fetch_url(url, check_cache_only=False, force_refresh=False):
    
    cache_file = map_url_to_cache_file(url)
    
    found_file = None
    for f in [cache_file, cache_file + '.gz']:
        if os.path.isfile(f):
            found_file = f
            break
    
    if (not check_cache_only 
        and (not found_file or force_refresh)):
        response = get_url(url)
        if not response:
            return None
        found_file = cache_file + '.gz'
        file_dir = os.path.dirname(found_file)
        try:
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
        except FileExistsError:
            # The crawler may run parallelized, so multiple threads/process 
            # instances may try to create it at the same time. Simply ignore
            # the error since it doesn't matter who creates it.
            pass
        with gzip.open(found_file, mode='wb') as f:
            f.write(response.text.encode())

    return found_file
    

def get_cache(url):
    
    found_file = fetch_url(url, check_cache_only=True)
    if not found_file:
        return None
    open_func = gzip.open if found_file.endswith('.gz') else open
    with open_func(found_file) as f:
        content = f.read().decode()
    return content


# In[4]:


def seed_cache(url):
    return (fetch_url(url) != None)

def parallel_fetch(url_list, num_workers=10):
    with mp.Pool(num_workers) as pool:
        results = pool.map(seed_cache, url_list)
    return len(list(filter(None, results)))

