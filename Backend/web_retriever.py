# -*- coding: utf-8 -*-
"""
Web Retriever
Handles web search and context retrieval from sources like Wikipedia and general web search.
"""
import requests
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ===== Web Retriever =====
class WebRetriever:
    """Web Retriever for RAG context"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

    def search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search Wikipedia using the opensearch API"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {'action': 'opensearch', 'search': query, 'limit': max_results, 'format': 'json'}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if len(data) < 4:
                return []
            titles, descriptions, urls = data[1], data[2], data[3]
            results = []
            for i in range(len(titles)):
                summary = self._get_wikipedia_summary(titles[i])
                results.append({
                    'source': 'Wikipedia',
                    'title': titles[i],
                    'url': urls[i],
                    'description': descriptions[i],
                    'content': summary
                })
            logger.info(f"✅ Wikipedia search completed, found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Wikipedia search failed: {str(e)}")
            return []

    def _get_wikipedia_summary(self, title: str, sentences: int = 5) -> str:
        """Get a summary of a Wikipedia page"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {'action': 'query', 'format': 'json', 'titles': title,
                      'prop': 'extracts', 'exintro': True, 'explaintext': True, 'exsentences': sentences}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    return page_data['extract']
            return ""
        except Exception as e:
            logger.error(f"Failed to get Wikipedia summary: {str(e)}")
            return ""

    def search_web(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Perform a general web search (using DuckDuckGo HTML search)"""
        try:
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query}
            response = self.session.post(url, data=data, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.find_all('div', class_='result', limit=max_results):
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                if title_elem and snippet_elem:
                    results.append({
                        'source': 'Web Search',
                        'title': title_elem.get_text(strip=True),
                        'url': title_elem.get('href', ''),
                        'description': snippet_elem.get_text(strip=True),
                        'content': snippet_elem.get_text(strip=True)
                    })
            logger.info(f"✅ Web search completed, found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []

    def retrieve_context(self, query: str, sources: List[str] = None) -> Dict[str, Any]:
        """Retrieve context from specified sources"""
        if sources is None:
            sources = ['wikipedia', 'web']
        logger.info(f"🔍 Starting retrieval for: {query}")
        all_results = []
        if 'wikipedia' in sources:
            all_results.extend(self.search_wikipedia(query))
        if 'web' in sources and len(all_results) < 3:
            all_results.extend(self.search_web(query))
        context_text = self._build_context_text(all_results)
        return {'query': query, 'results': all_results, 'context_text': context_text, 'total_sources': len(all_results)}

    def _build_context_text(self, results: List[Dict[str, str]]) -> str:
        """Build a single context string from search results"""
        if not results:
            return ""
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get('content', result.get('description', ''))
            if content:
                context_parts.append(f"[Source {i}] {content}")
        return " ".join(context_parts)