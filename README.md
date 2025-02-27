# Building a Support Agent Chatbot for CDP "How-to" Questions

This project is a Support Agent Chatbot designed to answer "how-to" questions related to Customer Data Platforms (CDPs) such as **Segment, mParticle, Lytics, and Zeotap**. The chatbot extracts relevant information from the official documentation, processes user queries using semantic search and summarization, and returns concise, accurate, and meaningful responses.

![Chatbot Demo](https://via.placeholder.com/600x200?text=Support+Agent+Chatbot)

## Features

- **How-to Questions:**  
  Provides step-by-step guidance for tasks such as setting up a new source in Segment or creating a user profile in mParticle.
  
- **Cross-CDP Comparisons:**  
  When a query involves multiple platforms (e.g., comparing Segment’s audience creation process to Lytics’), the chatbot returns a side‑by‑side comparison.

- **Advanced Query Handling:**  
  Supports complex queries with refined responses using a hybrid approach that combines predefined answers and dynamic semantic search.

- **Modern UI:**  
  A futuristic, dark-themed web interface with smooth animations and a loading indicator enhances user experience.

## Technologies & Libraries

- **Python 3.8+**
- **Flask:** For building the backend web application.
- **BeautifulSoup:** For scraping and parsing documentation content.
- **NLTK:** For tokenizing text.
- **SentenceTransformers:** For semantic search.
- **Transformers (Hugging Face):** For text summarization.
- **Torch (PyTorch):** For deep learning model support.

## Dependencies

Install the required dependencies using:

```bash
pip install flask requests beautifulsoup4 nltk sentence-transformers transformers torch
```


## Running the Chatbot

### Clone the Repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```
(Alternatively, run the pip command above.)

### Run the Flask Application:
```bash
python app.py
```

### Access the Chatbot:
Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Documentation Sources
The chatbot extracts data from these official documentation sites:
- [Segment Documentation](https://segment.com/docs/?ref=nav)
- [mParticle Documentation](https://docs.mparticle.com/)
- [Lytics Documentation](https://docs.lytics.com/)
- [Zeotap Documentation](https://docs.zeotap.com/home/en-us/)

---

## Performance & Specifications

### Fast Response:
Utilizes the lightweight `all-MiniLM-L6-v2` model for semantic search to ensure quick query processing.

### Efficient Summarization:
Employs the `sshleifer/distilbart-cnn-12-6` model for fast text summarization.

### Scalable Architecture:
Built with Flask and deployable using production-grade WSGI servers (e.g., Gunicorn) for high scalability.



## License
This project is licensed under the **MIT License**.


