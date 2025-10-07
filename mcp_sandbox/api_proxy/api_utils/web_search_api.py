import aiohttp
import asyncio
import traceback
import json
import os
import sys

# This part remains unchanged
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# This part remains unchanged
with open(os.path.join(current_dir, '../../configs/web_agent.json'), 'r') as f:
    config = json.load(f)


async def serper_google_search(
    query: str,
    serper_api_key: str = None,
    top_k: int = None,
    region: str = None,
    lang: str = None,
    depth: int = 0
):
    if serper_api_key is None:
        serper_api_key = config.get('serper_api_key')
    if top_k is None:
        top_k = config.get('top_k', 10)
    if region is None:
        region = config.get('search_region')
    if lang is None:
        lang = config.get('search_lang')

    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": top_k,
        "gl": region,
        "hl": lang,
        "location": "United States"
    }

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                data = await response.json()

                data.pop('searchParameters', None)
                data.pop('credits', None)

                if not data:
                    raise Exception("The google search API is temporarily unavailable, please try again later.")
                return data

    except Exception as e:
        if depth < 3:
            await asyncio.sleep(1)
            return await serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth + 1)
        print(f"search failed: {e}")
        print(traceback.format_exc())
        return []


async def serpapi_google_search(
    query: str,
    serper_api_key: str = None,
    top_k: int = None,
    region: str = None,
    lang: str = None,
    depth: int = 0
):
    if serper_api_key is None:
        serper_api_key = config.get('serper_api_key')
    if top_k is None:
        top_k = config.get('top_k', 10)
    if region is None:
        region = config.get('search_region')
    if lang is None:
        lang = config.get('search_lang')
    
    url = "https://serpapi.com/search" 
    params = {
        "q": query,
        "engine": "google",
        "api_key": serper_api_key,
        "num": top_k,
        "gl": region,
        "hl": lang,
        "location": "United States"
    }

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API Error: {response.status} - {error_text}")
                
                data = await response.json()

                data.pop('search_metadata', None)
                data.pop('search_parameters', None)
                data.pop('search_information', None)

                if not data.get('organic_results'):
                    raise Exception("The search API returned no organic results.")
                return data

    except Exception as e:
        if depth < 3:
            await asyncio.sleep(1)
            return await serpapi_google_search(query, serper_api_key, top_k, region, lang, depth=depth + 1)
        print(f"search failed: {e}")
        print(traceback.format_exc())
        return []



async def main():
    query = "[Merrill et al. Transformers are Hard-Attention Automata]"
    result = await serper_google_search(
        query=query,
        serper_api_key=config['serper_api_key'],
        top_k=10,
        region=config['search_region'],
        lang=config['search_lang']
    )
    print(json.dumps(result, indent=2))


# Script execution entry point. This part remains unchanged.
if __name__ == "__main__":
    asyncio.run(main())