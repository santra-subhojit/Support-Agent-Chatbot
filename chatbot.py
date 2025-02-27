from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import numpy as np
import re
import time

nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

def clean_text(text):
    """
    Removes common navigation phrases and extra whitespace.
    Extend this function as needed.
    """
    unwanted = [
        r'View More', r'Overview', r'Index', r'Accessing Accounts',
        r'Learn More', r'Getting Started', r'Resources', r'\bSpec\b',
        r'\bGuide\b', r'\bInstallation\b'
    ]
    for pattern in unwanted:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def refine_answer(text, max_words=50):
    """
    If the text is longer than max_words, use the summarization pipeline
    to generate a refined, concise answer.
    """
    words = text.split()
    if len(words) > max_words:
        summary = summarizer(text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    return text

def is_relevant_sentence(sentence, keywords, min_words=5, max_words=80):
    """
    Returns True if the sentence is within the desired length and contains at least one relevant keyword.
    """
    words = sentence.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    for kw in keywords:
        if kw.lower() in sentence.lower():
            return True
    return False

def filter_candidates_by_keywords(candidates, keywords):
    """
    Filters candidate sentences to include only those that meet the length requirements
    and contain at least one of the given keywords.
    """
    return [candidate for candidate in candidates if is_relevant_sentence(candidate, keywords)]

def get_predefined_answer(query):
    """
    Checks if the query matches a known pattern and returns a predefined answer.
    """
    nq = query.lower().strip()
    if "set up a new source in segment" in nq or "new source in segment" in nq:
        return (
            "Segment: To set up a new source, log in to your Segment account, navigate to the Sources tab, "
            "click 'Add Source', and follow the on‑screen instructions to configure your data tracking."
        )
    elif "create a user profile in mparticle" in nq or "user profile in mparticle" in nq:
        return (
            "mParticle: To create a user profile, send an Identify call with the user's traits. "
            "mParticle uses this data to build or update the user profile automatically."
        )

    return None

def fetch_documentation(url):
    """
    Fetches HTML content from the URL and extracts main text.
    Adjust element selection based on the site's structure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/105.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('main')
            if main_content:
                return main_content.get_text(separator=" ", strip=True)
            else:
                return soup.get_text(separator=" ", strip=True)
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

DOC_URLS = {
    'segment': 'https://segment.com/docs/?ref=nav',
    'mparticle': 'https://docs.mparticle.com/',
    'lytics': 'https://docs.lytics.com/',
    'zeotap': 'https://docs.zeotap.com/home/en-us/'
}

def load_documentation_from_urls(urls):
    """
    Fetches and returns documentation text for each platform.
    """
    documents = {}
    for platform, url in urls.items():
        print(f"Fetching documentation for {platform}...")
        time.sleep(1)
        doc_text = fetch_documentation(url)
        documents[platform] = doc_text
    return documents

def preprocess_docs(documents):
    """
    Splits documentation text into individual sentences.
    Returns a list of sentences and a parallel list of platform labels.
    """
    sentences = []
    labels = []
    for platform, text in documents.items():
        sents = sent_tokenize(text)
        sentences.extend(sents)
        labels.extend([platform] * len(sents))
    return sentences, labels

print("Loading documentation and initializing models...")
documents = load_documentation_from_urls(DOC_URLS)
sentences, labels = preprocess_docs(documents)

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

KNOWN_PLATFORMS = ['segment', 'mparticle', 'lytics', 'zeotap']

RELEVANT_KEYWORDS = ["audience", "segment", "creation", "process", "profile", "workflow", "tracking", "source", "setup", "identify"]

def detect_platforms(query):
    """Detects which platforms are mentioned in the query."""
    query_lower = query.lower()
    return [platform for platform in KNOWN_PLATFORMS if platform in query_lower]

def is_comparison_query(query):
    """Determines if the query includes comparison keywords."""
    comparison_keywords = ["compare", "versus", "vs", "difference"]
    return any(keyword in query.lower() for keyword in comparison_keywords)

def is_advanced_query(query):
    """Determines if the query includes advanced keywords."""
    advanced_keywords = ["advanced", "integration", "integrate", "configuration", "config", "use case"]
    return any(keyword in query.lower() for keyword in advanced_keywords)

def summarize_candidates(candidates):
    """
    If multiple candidate sentences exist, concatenate and summarize them.
    Otherwise, return the cleaned candidate text.
    """
    if not candidates:
        return ""
    combined_text = " ".join(candidates)
    combined_text = clean_text(combined_text)
    if len(combined_text.split()) < 40:
        return combined_text
    summary = summarizer(combined_text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def get_response(query, top_n=3):
    """
    Processes the user's query using semantic search.
    First checks for predefined answers for known queries.
    For comparison queries, selects the best candidate per platform (after filtering by keywords)
    and refines the answer.
    For default queries, selects top candidates (filtered for relevance) and refines the answer.
    """
   
    predefined = get_predefined_answer(query)
    if predefined:
        return predefined

    if is_advanced_query(query):
        top_n = 5

    platforms_in_query = detect_platforms(query)
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Branch for cross-CDP comparison queries
    if is_comparison_query(query) and len(platforms_in_query) >= 2:
        responses = []
        comparison_threshold = 0.05
        for platform in platforms_in_query:
            indices = [i for i, p in enumerate(labels) if p == platform]
            if not indices:
                continue
            platform_sentences = [sentences[i] for i in indices]
            platform_embeddings = model.encode(platform_sentences, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, platform_embeddings)[0]
            if cosine_scores.max().item() < comparison_threshold:
                best_candidate = "No relevant information found."
            else:
                best_index = int(torch.argmax(cosine_scores))
                best_candidate = clean_text(platform_sentences[best_index])
            filtered = filter_candidates_by_keywords([best_candidate], RELEVANT_KEYWORDS)
            best_candidate = filtered[0] if filtered else best_candidate
            refined = refine_answer(best_candidate)
            responses.append(f"{platform.capitalize()}: {refined}")
        return "\n\n".join(responses)
    else:
        cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        similarity_threshold = 0.1
        if cosine_scores.max().item() < similarity_threshold:
            return ("It appears your question may not be related to our supported CDP topics. "
                    "Please ask a question about how to use features in Segment, mParticle, Lytics, or Zeotap.")
        cosine_scores_np = cosine_scores.cpu().numpy()
        top_indices = np.argpartition(-cosine_scores_np, range(top_n))[:top_n]
        top_indices = sorted(top_indices, key=lambda i: -cosine_scores_np[i])
        candidates = [clean_text(sentences[i]) for i in top_indices]
        filtered_candidates = filter_candidates_by_keywords(candidates, RELEVANT_KEYWORDS)
        final_candidates = filtered_candidates if filtered_candidates else candidates
        answer = summarize_candidates(final_candidates) if len(final_candidates) > 1 else final_candidates[0]
        refined = refine_answer(answer)
        return refined

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided."}), 400
    response_text = get_response(query)
    return jsonify({"response": response_text})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
